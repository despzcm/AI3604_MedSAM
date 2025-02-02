# 🫁Medical-sam ![Static Badge](https://img.shields.io/badge/Python-blue)
## 🕹️Introduction
> Image segmentation, a key challenge in computer vision, involves dividing an image into distinct, non-overlapping regions based on features like color, texture, and shape. It is crucial for image analysis and understanding, separating the target object from the background. Since the 1990s, segmentation has received significant research attention, with deep learning advancements, especially models like Segment Anything Model (SAM), driving substantial progress in this area.

> In this project, we implement medical image segmentation using the `SAM` model and 
fine-tune SAM using efficient fine-tuning techniques (e.g., `LoRA`, `Adapters`) to fit the specific 
requirements of the medical imaging dataset.

## 🫀Sample Results
![Sample](/stacked.png)

## 🚀Evaluate
main file: `model_eval.py`
```
python model_eval.py    --data_path $YOUR_DATASET_PATH\
                        --sam_model_path $YOUR_SAM_MODEL_PATH\
                        --medsam_model_path $YOUR_MedSAM_MODEL_PATH\
                        --ckp_path $YOUR_CHECKPOINT_PATH\
                   
```

## 🤖Our model
> We use [SAM](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://github.com/bowang-lab/MedSAM) as our base model.

[Model Link](https://jbox.sjtu.edu.cn/l/tH1JU0)

* `medsam_base` MedSAM base model
* `medsam_lora` MedSAM finetune use lora
* `medsam_boft` MedSAM finetune use boft
* `sam_base` SAM base model
* `sam_lora` SAM finetune use lora
* `sam_boft` SAM finetune use boft


## 🔬Dataset and Data Process
> For the training dataset, we primarily use 80\% of the labeled `FLARE22 dataset`, which consists of approximately 3,600 images after preprocessing. For evaluation, we employ three distinct datasets. These include `FLARE22`, a large-scale and diverse abdomen CT dataset from over 20 medical institutions, with the remaining labeled data used for evaluation; the `ETIS-Larib Polyp DB`, a database containing frames extracted from colonoscopy videos, with 192 labeled polyp images; and `SegTHOR`, a CT dataset specifically designed for the segmentation of thoracic organs at risk (OARs), where we use 400 labeled images for testing.

`npy_img_dataset.py` our new data class

`flare22_process.py` process [Flare22 dataset](https://flare22.grand-challenge.org/Dataset/)

`SegTHOR_process.py` process [SegTHOR dataset](https://competitions.codalab.org/competitions/21145)

Part of dataset we use: [Link](https://jbox.sjtu.edu.cn/l/x1lUuz)

## 🛠️Model Train
`finetune_lora_medsam.py` use lora to finetune medsam

`finetune_lora_SAM.py` use lora to finetune sam

`adapter_medsam.py` use adapter to finetune medsam

`adapter_SAM.py` use adapter to finetune sam



## More Datasets
* MICCAI FLARE22 Challenge Dataset        MICCAI FLARE 2023        https://github.com/JunMa11/FLARE
* MPLiTS
* MPCT-FLLs
* https://github.com/linhandev/dataset
* saros：https://www.cancerimagingarchive.net/analysis-result/saros/
* brain-tumor-mri： https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
* https://blog.csdn.net/qq_40714949/article/details/125979508 Polyp Segmentation
* The ISIC 2020 Challenge Dataset：https://challenge2020.isic-archive.com/








