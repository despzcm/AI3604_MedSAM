# Medical-sam ![Static Badge](https://img.shields.io/badge/Python-blue)
## Introduction
> Image segmentation, a key challenge in computer vision, involves dividing an image into distinct, non-overlapping regions based on features like color, texture, and shape. It is crucial for image analysis and understanding, separating the target object from the background. Since the 1990s, segmentation has received significant research attention, with deep learning advancements, especially models like Segment Anything Model (SAM), driving substantial progress in this area.

> In this project, we implement medical image segmentation using the `SAM` model and 
fine-tune SAM using efficient fine-tuning techniques (e.g., `LoRA`, `Adapters`) to fit the specific 
requirements of the medical imaging dataset. 
## Data Process
`npy_img_dataset.py` our new data class

`flare22_process.py` process Flare22 dataset

`SegTHOR_process.py` process SegTHOR dataset

## Model Train
`finetune_lora_medsam.py` use lora to finetune medsam

`finetune_lora_SAM.py` use lora to finetune sam

`adapter_medsam.py` use adapter to finetune medsam

`adapter_SAM.py` use adapter to finetune sam

## Model Eval
main file: model_eval.py
```
python model_eval.py    --data_path $YOUR_DATASET_PATH\
                        --sam_model_path $YOUR_SAM_MODEL_PATH\
                        --medsam_model_path $YOUR_MedSAM_MODEL_PATH\
                        --ckp_path $YOUR_CHECKPOINT_PATH\
                   
```

## dataset
* MICCAI FLARE22 Challenge Dataset        MICCAI FLARE 2023        https://github.com/JunMa11/FLARE
* MPLiTS
* MPCT-FLLs
* 包含一切的数据集：https://github.com/linhandev/dataset 但是问题在于比较老
* 一些可能不是医学的数据集： https://github.com/tianrun-chen/SAM-Adapter-PyTorch 链接中的dataset
* saros：https://www.cancerimagingarchive.net/analysis-result/saros/
* brain-tumor-mri： https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
* https://blog.csdn.net/qq_40714949/article/details/125979508 Polyp Segmentation所划分好的训练集与测试集
* The ISIC 2020 Challenge Dataset：https://challenge2020.isic-archive.com/
* https://jbox.sjtu.edu.cn/l/x1lUuz  demo dataset
* https://github.com/gregoryperkins/MTL-SegTHOR?tab=readme-ov-file SegTHOR dataset.疑似能用 .nii, data and the process code: https://jbox.sjtu.edu.cn/l/L1Kwd8


## Accuracy of SAM
* 论文：https://arxiv.org/pdf/2304.09324v1 里面也包括了一些数据集
* 同时也可以作为作图的参考
* https://arxiv.org/pdf/2304.12306v1 这篇有比较多的数据对比，以及具体的评估方法，但是没有代码

## model usage
* huggingface上可以直接用的模型，微调也比较方便 https://huggingface.co/flaviagiammarino/medsam-vit-base   https://huggingface.co/wanglab/medsam-vit-base



