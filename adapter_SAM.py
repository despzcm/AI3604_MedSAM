from peft import LoraConfig, get_peft_model, IA3Config, BOFTConfig
from peft import PeftConfig
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import random
from datetime import datetime
import shutil
from segment_anything import sam_model_registry
import glob
import monai
import torch
import torch.nn as nn
from tqdm import tqdm
# from train_one_gpu import MedSAM
from torch.optim import AdamW
import torch.nn.functional as F

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of low-rank adapters (you can experiment with this)
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout to regularize LoRA layers
    bias="none",  # Bias handling in LoRA layers (can be 'none', 'all', 'lora', etc.)
    target_modules=["q_proj", "v_proj"]  # Specify which layers to apply LoRA to (for example)
)
ia_config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"], 
    feedforward_modules=["down_proj"]
)


boft_config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    # target_modules=["mask_decoder.output_hypernetworks_mlps.0.layers.2",
    #                 "mask_decoder.output_hypernetworks_mlps.1.layers.2",
    #                 "mask_decoder.output_hypernetworks_mlps.2.layers.2",
    #                 "mask_decoder.output_hypernetworks_mlps.3.layers.2",
    #                 "mask_decoder.transformer.layers.1.mlp.lin1",
    #                 "mask_decoder.transformer.layers.1.mlp.lin2",
    #                 "mask_decoder.transformer.layers.0.mlp.lin1",
    #                 "mask_decoder.transformer.layers.0.mlp.lin2"],
    
    target_modules=["mask_decoder.output_hypernetworks_mlps.0.layers.2",
                    "mask_decoder.output_hypernetworks_mlps.1.layers.2",
                    "mask_decoder.output_hypernetworks_mlps.2.layers.2",
                    "mask_decoder.output_hypernetworks_mlps.3.layers.2",
                    # "mask_decoder.transformer.layers.1.mlp.lin1",
                    # "mask_decoder.transformer.layers.1.mlp.lin2",
                    # "mask_decoder.transformer.layers.0.mlp.lin1",
                    # "mask_decoder.transformer.layers.0.mlp.lin2",
                   "image_encoder.blocks.0.mlp.lin2",
                   "image_encoder.blocks.1.mlp.lin2",
                   "image_encoder.blocks.2.mlp.lin2",
                   "image_encoder.blocks.3.mlp.lin2",
                   "image_encoder.blocks.4.mlp.lin2",
                   "image_encoder.blocks.5.mlp.lin2",
                   "image_encoder.blocks.6.mlp.lin2",
                   "image_encoder.blocks.7.mlp.lin2",
                   "image_encoder.blocks.8.mlp.lin2",
                   "image_encoder.blocks.9.mlp.lin2",
                   "image_encoder.blocks.10.mlp.lin2",
                   "image_encoder.blocks.11.mlp.lin2"],
    
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

device = "cuda:1"
sam_model = sam_model_registry["vit_b"](checkpoint="./checkpoint/sam_vit_b_01ec64.pth")
sam_model.to(device)
#for name, module in sam_model.named_modules():
    #print(name)
npy_dir = "./data/npy/CT_Abd"
num_epochs = 3
set_num_workers = 32
set_batch_size = 2
learning_rate = 2e-5

model = get_peft_model(sam_model, boft_config)
optimizer = AdamW(model.parameters(), lr=learning_rate)

from segment_anything.utils.transforms import ResizeLongestSide
resize_transform = ResizeLongestSide(model.image_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device = device) 
    return image.contiguous()
    

#predictor = SamPredictor(model)

#checkpoint = torch.load(MedSAM_CKPT_PATH, map_location=device)
#print(checkpoint.keys())
#model.load_state_dict(checkpoint)

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            os.path.join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        #print("img_shape",img_1024.shape)
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        #print("img_shape0",img_1024.shape)
        #print("0",torch.tensor(img_1024).float().size())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )
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


# Example dataset loading (adjust paths accordingly)
train_dataset = NpyDataset(npy_dir)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=set_batch_size,
    shuffle=True,
    num_workers=set_num_workers,
    pin_memory=False,
)



# Define loss function
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

def test_original(model, dataloader):
    print("original test")
    model.eval()
    total_loss = 0
    for step, (image, gt2D, boxes, _) in enumerate(tqdm(dataloader)):
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
        
        loss = seg_loss(masks_torch.float(), gt2D.float()) + ce_loss(masks_torch.float(), gt2D.float())
        total_loss += loss.item()
    return total_loss / len(dataloader)

# original_model_loss = test_original(sam_model,train_dataloader)
# print("original loss", original_model_loss)


# Define the training loop
def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for step, (image, gt2D, boxes, _) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
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
        batched_output = model(batched_input, multimask_output=False)
        
        masks_list = [entry["masks"] for entry in batched_output]
        masks_torch = torch.cat([entry for entry in masks_list], dim=0).to(device).float()
        
        loss = seg_loss(masks_torch.float(), gt2D.float()) + ce_loss(masks_torch.float(), gt2D.float())
         
        # Backpropagation
        loss.backward()

        #for entry in batched_output:
            #print(batched_output[0]["masks"].requires_grad)  # Should be True
            #print(batched_output[0]["masks"].grad_fn)        # Should not be None
        
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

for epoch in range(num_epochs):
    epoch_loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch+1,
    }
    torch.save(checkpoint, "./checkpoint/finetune_sam_boft_adapter"+ str(epoch+1)+".pth")

