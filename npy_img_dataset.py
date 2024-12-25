from torch.utils.data import Dataset, DataLoader
import torch
import os
import glob
import numpy as np
import random
from PIL import Image,ImageDraw

#当前仅适用于单色图片
class NpyImgDataset(Dataset):
    def __init__(self, npy_data_dir:str=None, img_data_dir:str=None, img_format:str='jpg',bbox_shift:int=20):
        print("Initializing NpyDataset...")
        if(npy_data_dir is None and img_data_dir is None):
            raise ValueError("No data.")
        self.img_format = img_format
        self.bbox_shift = bbox_shift
        
        self.npy_data_dir = npy_data_dir
        self.img_data_dir = img_data_dir
        self.npy_gt_paths=None
        self.img_gt_paths=None
        if(npy_data_dir is not None):
            self.npy_gt_paths = sorted(
                glob.glob(os.path.join(self.npy_data_dir,"gts", "**/*.npy"), recursive=True)
            )
            self.npy_gt_paths = [
                file
                for file in self.npy_gt_paths
                if os.path.isfile(os.path.join(self.npy_data_dir,"imgs", os.path.basename(file)))
            ]
        if(img_data_dir is not None):
            self.img_gt_paths = sorted(
                glob.glob(os.path.join(self.img_data_dir,"gts", "**/*."+self.img_format), recursive=True)
            )
            self.img_gt_paths = [
                file
                for file in self.img_gt_paths
                if os.path.isfile(os.path.join(self.img_data_dir,"imgs", os.path.basename(file)))
            ]
        if(self.npy_gt_paths is None and self.img_gt_paths is None):
            raise ValueError("No data found.")
        
        print((len(self.npy_gt_paths) if self.npy_gt_paths is not None else 0),'npy data')
        print((len(self.img_gt_paths)if self.img_gt_paths is not None else 0),'img data')
    def __len__(self):
        return (len(self.npy_gt_paths) if self.npy_gt_paths is not None else 0)+(len(self.img_gt_paths)if self.img_gt_paths is not None else 0)
    def __getitem__(self, index):
        img_name = None
        data_dir=None
        if self.npy_gt_paths is not None and self.img_gt_paths is not None:
            if index>=len(self.npy_gt_paths):
                index-=len(self.npy_gt_paths)
                img_name = os.path.basename(self.img_gt_paths[index])
                data_dir = self.img_data_dir
            else:
                img_name = os.path.basename(self.npy_gt_paths[index])
                data_dir = self.npy_data_dir
        elif self.npy_gt_paths is not None:
            img_name = os.path.basename(self.npy_gt_paths[index])
            data_dir = self.npy_data_dir
        elif self.img_gt_paths is not None:
            img_name = os.path.basename(self.img_gt_paths[index])
            data_dir = self.img_data_dir
        else:
            #unlikely to reach here
            raise ValueError("No data found.")
        if data_dir == self.npy_data_dir:
            img_1024 = np.load(
                os.path.join(data_dir, "imgs", img_name), "r", allow_pickle=True
            )# (1024, 1024, 3)
            img_1024 = np.transpose(img_1024, (2, 0, 1))
            gt = np.load(
                self.npy_gt_paths[index], "r", allow_pickle=True
            )  # multiple labels [0, 1,4,5...], (256,256)
    
            label_ids = np.unique(gt)[1:]#除去0
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
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
            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt2D[None, :, :]).long(),
                torch.tensor(bboxes).float(),
                img_name,
            )
        elif data_dir == self.img_data_dir:
            img = np.array(Image.open(
                os.path.join(data_dir, "imgs", img_name),'r'
            ).resize((1024,1024)),dtype=np.float32)/255
            img = np.transpose(img, (2, 0, 1))
            
            gt  = np.array((Image.open(
                os.path.join(data_dir, "gts", img_name),'r'
            ).resize((1024,1024),Image.Resampling.NEAREST).convert('L')),dtype=np.uint8)
            
            #由于jpg图像可能使得图像二值化失去锐边，因此使用阈值128，注意多标签情况
            if self.img_format=='jpg':
                gt[gt>=128]=255
                gt[gt<128]=0
            label_ids = np.unique(gt)[1:]
            # print(label_ids)
            
            gt2d=np.uint8(gt==random.choice(label_ids.tolist()))
            y_indices, x_indices = np.where(gt2d > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2d.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
            return (
                torch.tensor(img).float(),
                torch.tensor(gt2d[None, :, :]).long(),
                torch.tensor(bboxes).float(),
                img_name,
            )
        else:
            #unlikely to reach here
            raise ValueError("No data found.")

# npy_dir='./datasets/npy_test'
# img_dir='./datasets/Kvasir-SEG'
# train_dataset = NpyDataset(npy_dir,img_dir)


# img,gt2d,bbox,imgname=train_dataset[2]
# print(img.shape)
# print(gt2d.shape)
# print(bbox)
# print(imgname)
# imgg=Image.fromarray(np.transpose(img.numpy(),(1,2,0)).astype(np.uint8))
# draw = ImageDraw.Draw(imgg)
# draw.rectangle(bbox.tolist(), outline="red")
# imgg.show()
