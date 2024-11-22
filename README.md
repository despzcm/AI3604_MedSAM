# Medical-sam

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


## Accuracy of SAM
* 论文：https://arxiv.org/pdf/2304.09324v1 里面也包括了一些数据集
* 同时也可以作为作图的参考

## model usage
* huggingface上可以直接用的模型，微调也比较方便 https://huggingface.co/flaviagiammarino/medsam-vit-base   https://huggingface.co/wanglab/medsam-vit-base

## To-DO list
* 数据预处理，对于不同数据集处理到同样的格式  目前遇到的问题：预处理速度非常之慢
* 训练微调代码
* 如何在服务器上训练
* 测试模型的精度，不同的评估办法
