import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage import measure
from scipy.spatial.distance import cdist
import glob
import os
import random
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.nn as nn
from peft import LoraConfig, get_peft_model, BOFTConfig, IA3Config
from MedSAM.segment_anything.utils.transforms import ResizeLongestSide
from npy_img_dataset import NpyImgDataset
from scipy.spatial import cKDTree
from monai.metrics import compute_surface_dice,SurfaceDiceMetric
import argparse
join = os.path.join

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./SegTHOR/CT_Abd")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--model_type", type=str, default="lora", help="lora, boft, ia, medsam")
parser.add_argument("--sam_model_path", type=str, default="./checkpoint/sam_vit_b_01ec64.pth")
parser.add_argument("--medsam_model_path", type=str, default="./checkpoint/medsam_vit_b.pth")
parser.add_argument("--ckp_path", type=str, default="./checkpoint/fintune_medsam_lora_epoch3.pth")

args = parser.parse_args()


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


lora_config = LoraConfig(
    r=8,  # Rank of low-rank adapters (you can experiment with this)
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout to regularize LoRA layers
    bias="none",  # Bias handling in LoRA layers (can be 'none', 'all', 'lora', etc.)
    target_modules=["q_proj", "v_proj"]  # Specify which layers to apply LoRA to (for example)
)

boft_config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    target_modules=["mask_decoder.output_hypernetworks_mlps.0.layers.2",
                    "mask_decoder.output_hypernetworks_mlps.1.layers.2",
                    "mask_decoder.output_hypernetworks_mlps.2.layers.2",
                    "mask_decoder.output_hypernetworks_mlps.3.layers.2",
                    "mask_decoder.transformer.layers.1.mlp.lin1",
                    "mask_decoder.transformer.layers.1.mlp.lin2",
                    "mask_decoder.transformer.layers.0.mlp.lin1",
                    "mask_decoder.transformer.layers.0.mlp.lin2"],
    
    # target_modules=["mask_decoder.output_hypernetworks_mlps.0.layers.2",
    #                 "mask_decoder.output_hypernetworks_mlps.1.layers.2",
    #                 "mask_decoder.output_hypernetworks_mlps.2.layers.2",
    #                 "mask_decoder.output_hypernetworks_mlps.3.layers.2",
    #                "image_encoder.blocks.0.mlp.lin2",
    #                "image_encoder.blocks.1.mlp.lin2",
    #                "image_encoder.blocks.2.mlp.lin2",
    #                "image_encoder.blocks.3.mlp.lin2",
    #                "image_encoder.blocks.4.mlp.lin2",
    #                "image_encoder.blocks.5.mlp.lin2",
    #                "image_encoder.blocks.6.mlp.lin2",
    #                "image_encoder.blocks.7.mlp.lin2",
    #                "image_encoder.blocks.8.mlp.lin2",
    #                "image_encoder.blocks.9.mlp.lin2",
    #                "image_encoder.blocks.10.mlp.lin2",
    #                "image_encoder.blocks.11.mlp.lin2"],
    
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

ia_config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"], 
    feedforward_modules=["down_proj"]
)

batch_size=args.batch_size


def dice_coefficient(pred_mask, gt_mask):
    # print(pred_mask.shape, gt_mask.shape)
    smooth = 1e-6
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    return 2 * intersection / (union + smooth)


def compute_nsd_batch(y_true_batch, y_pred_batch):
    nsd_scores = 0
    
    for i in range(y_true_batch.shape[0]):
        y_true = y_true_batch[i]
        y_pred = y_pred_batch[i]
        if y_true.ndim == 2 and y_pred.ndim == 2:
            contours_true = measure.find_contours(y_true, 0.5)
            contours_pred = measure.find_contours(y_pred, 0.5)
            
            contour_true_points = np.vstack(contours_true)
            contour_pred_points = np.vstack(contours_pred)

            dist_true_to_pred = cdist(contour_true_points, contour_pred_points, 'euclidean')
            D_front = np.min(dist_true_to_pred, axis=1).mean()

            dist_pred_to_true = cdist(contour_pred_points, contour_true_points, 'euclidean')
            D_back = np.min(dist_pred_to_true, axis=1).mean()

            A_front = len(contour_pred_points)
            A_back = len(contour_true_points)

            nsd = (D_front + D_back) / max(A_front, A_back)
            nsd_scores += nsd
        else:
            contours_true = measure.find_contours(y_true_batch, 0.5)
            contours_pred = measure.find_contours(y_pred_batch, 0.5)
            contour_true_points = np.vstack(contours_true)
            contour_pred_points = np.vstack(contours_pred)

            dist_true_to_pred = cdist(contour_true_points, contour_pred_points, 'euclidean')
            D_front = np.min(dist_true_to_pred, axis=1).mean()

            dist_pred_to_true = cdist(contour_pred_points, contour_true_points, 'euclidean')
            D_back = np.min(dist_pred_to_true, axis=1).mean()

            A_front = len(contour_pred_points)
            A_back = len(contour_true_points)

            nsd = (D_front + D_back) / max(A_front, A_back)
            nsd_scores += nsd
            return nsd_scores/batch_size
        
    return nsd_scores/y_true_batch.shape[0]

def evaluate(model, test_dataloader, device):
    model.eval()
    total_dice_score = 0
    total_nsd_score = 0 
    num_samples = len(test_dataloader.dataset)
    with torch.no_grad():
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(test_dataloader)):
            image, gt2D = image.to(device), gt2D.to(device)
            boxes_np = boxes.detach().cpu().numpy()
            medsam_pred = model(image, boxes_np)
            medsam_pred=torch.sigmoid(medsam_pred)
            
            medsam_pred = (medsam_pred > 0.5).float()
            nsd_score = compute_surface_dice(medsam_pred, gt2D,class_thresholds=[10])
            
            medsam_pred=medsam_pred.cpu().numpy()
            medsam_pred=medsam_pred.squeeze()
            gt2D=gt2D.cpu().numpy()
            gt2D=gt2D.squeeze()
            
            dice_score = dice_coefficient(medsam_pred, gt2D)
            total_dice_score += dice_score
            nsd_score=torch.mean(nsd_score)
            total_nsd_score += nsd_score.item()

    avg_dice_score = total_dice_score / num_samples*batch_size
    avg_nsd_score = total_nsd_score / num_samples*batch_size
    return avg_dice_score, avg_nsd_score

def prepare_image(image, transform, device):
    image = torch.as_tensor(image, device = device) 
    return image.contiguous()


def evaluate_sam(model, test_dataloader, device):
    model.eval()
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)
    total_dice_score = 0
    total_nsd_score = 0
    for step, (image, gt2D, boxes, _) in enumerate(tqdm(test_dataloader)):
        boxes_np = boxes.detach().to(device)
        image, gt2D = image.to(device), gt2D.to(device)
        batched_input = []
        for i in range(image.shape[0]):
            batched_input.append({
                 'image': prepare_image(image[i], resize_transform, device).float(),
                 'boxes': resize_transform.apply_boxes_torch(boxes_np[i], image[i].shape[-2:]),
                 'original_size': image[i].shape[-2:]
             })
        # Forward pass
        with torch.no_grad():
            batched_output = model(batched_input, multimask_output=False)
        
        masks_list = [entry["masks"] for entry in batched_output]
        masks_torch = torch.cat([entry for entry in masks_list], dim=0).to(device).float()
        
        masks_torch=torch.sigmoid(masks_torch)
        masks_torch = (masks_torch > 0.5).float()
        
        nsd_score = compute_surface_dice(masks_torch, gt2D,class_thresholds=[10])
        # print(nsd_score)
        masks_torch=masks_torch.cpu().numpy()
        masks_torch=masks_torch.squeeze()
        gt2D=gt2D.cpu().numpy()
        gt2D=gt2D.squeeze()
        
        # print(masks_torch.shape, gt2D.shape)
        dice_score = dice_coefficient(masks_torch, gt2D)    
        total_dice_score += dice_score
        nsd_score=torch.mean(nsd_score)
        # print(nsd_score)
        total_nsd_score += nsd_score.item()
    avg_dice_score = total_dice_score / len(test_dataloader)
    avg_nsd_score = total_nsd_score / len(test_dataloader)
    return avg_dice_score, avg_nsd_score
        

def sam_main(model_type="sam"):   
    model = sam_model_registry["vit_b"](checkpoint=args.sam_model_path)
    model.to(device)
    if model_type == "lora":
        model = get_peft_model(model, lora_config)
        checkpoint = torch.load(args.ckp_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == "boft":    
        model=get_peft_model(model, boft_config)
        checkpoint = torch.load(args.ckp_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == "ia":
        model=get_peft_model(model, ia_config)
        checkpoint = torch.load(args.ckp_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        
    print("Model loaded successfully")
    # avg_dice_score,avg_nsd_score = evaluate(medsam_model, test_dataloader, device)    
    
    
    
    # avg_dice_score,avg_nsd_score=evaluate_sam(model, test_dataloader, device)
    
    # avg_nsd_score=avg_nsd_score[0]
    avg_dice_score,avg_nsd_score=evaluate_sam(model, test_dataloader, device)
    print(f"Final Average Dice Score on Test Set: {avg_dice_score:.4f}")
    print(f"Final Average NSD Score on Test Set: {avg_nsd_score:.4f}")



def medsam_main(model_type="medsam"):
    
    # model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base",cache_dir="model").to(device)
    # processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base",cache_dir="model")
    
    sam_model = sam_model_registry["vit_b"](checkpoint=args.medsam_model_path)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    
    if model_type == "lora":
        MedSAM_CKPT_PATH=args.ckp_path
        model = get_peft_model(medsam_model, lora_config)
        checkpoint = torch.load(MedSAM_CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == "boft":
        MedSAM_CKPT_PATH=args.ckp_path
        checkpoint = torch.load(MedSAM_CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
    elif model_type=='medsam':
        model=medsam_model
        MedSAM_CKPT_PATH=args.medsam_model_path
        checkpoint = torch.load(MedSAM_CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("Model loaded successfully")
    # avg_dice_score,avg_nsd_score = test(model, processor, test_dataloader, device)
    
    avg_dice_score,avg_nsd_score = evaluate(model, test_dataloader, device)    
    
    print(f"Final Average Dice Score on Test Set: {avg_dice_score:.4f}")
    print(f"Final Average NSD Score on Test Set: {avg_nsd_score:.4f}")

if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    test_dataset = NpyImgDataset(
        npy_data_dir=args.data_path,
    )  
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=32) 
       
    sam_main(model_type=args.model_type)
    medsam_main(model_type=args.model_type)
    