# -*- coding: utf-8 -*-
# %% import packages
import numpy as np
import cv2
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d
import operator


img_name_suffix = ".nii"  # Changed from .nii.gz to .nii
gt_name_suffix = ".nii"  # Changed from .nii.gz to .nii
modality = "CT"
anatomy = "Abd"  # Anatomy + dataset name
prefix = modality + "_" + anatomy + "_"

nii_path = "./data/SegTHOR/images"  # Path to the .nii images
gt_path = "./data/SegTHOR/labels"  # Path to the ground truth
npy_path = "./data/SegTHOR"
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)

image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"Original number of files: {len(names)}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"Number of files after sanity check: {len(names)}")

# Set label IDs to exclude
remove_label_ids = []  # Remove duodenum since it is scattered in the image, making it hard to specify with a bounding box
tumor_id = None  # Only set this when there are multiple tumors; convert semantic masks to instance masks

# Set window level and width for CT images
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400

# %% Save preprocessed images and masks as .npz files
for name in tqdm(names):  # Use the remaining cases for validation
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # Remove unwanted labels
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # Handle tumor masks (if specified)
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )
    # Remove small objects in 3D and 2D
    gt_data_ori = cc3d.dust(gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True)
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        gt_data_ori[slice_i, :, :] = cc3d.dust(gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True)
    # Find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)
    if len(z_index) > 0:
        # Crop the ground truth and load image
        gt_roi = gt_data_ori[z_index, :, :]
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # Preprocess image
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0
        
        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + '.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # Save processed .nii files
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii"),
        )
        # Save resized slices as .npy
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            resize_img = cv2.resize(img_3c, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            gt_i = gt_roi[i, :, :]
            resize_gt = cv2.resize(gt_i, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            resize_gt = np.uint8(resize_gt)
            assert resize_img.shape[:2] == resize_gt.shape
            print(np.max(resize_img))
            print(np.max(resize_gt))
            np.save(join(npy_path, "imgs", f"{prefix}{gt_name.split(gt_name_suffix)[0]}-{i:03d}.npy"), resize_img)
            np.save(join(npy_path, "gts", f"{prefix}{gt_name.split(gt_name_suffix)[0]}-{i:03d}.npy"), resize_gt)
