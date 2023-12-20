# A Cross-Domain Challenge with Panoptic Segmentation in Agriculture

In this github repository we release the novel WeedAI dataset and our pseudo-labels and train, validation, evaluation splits associated with the Crop and Weed dataset (CAW) [1].

> Michael Halstead, Patrick Zimmer, and Chris McCool.
> **A Cross-Domain Challenge with Panoptic Segmentation in Agriculture**
> _The International Journal of Robotics Research_


## WeedAI dataset

The WeedAI dataset is available [here](https://uni-bonn.sciebo.de/s/vJBrCahb4lvDSXQ) for download. It is available in the CoCo format and an example dataloader is available in the libs directory of this repository.


To use this dataloader you will need the following requirements (see _requirements.txt_):
- numpy==1.19.5
- torch==1.9.1
- torchvision==0.10.1
- torchcontrib==0.0.2
- pytorch-lightning==1.2.10
- scikit-learn==0.24.2
- scikit-image==0.17.2
- tensorboardX
- opencv-python==4.5.3.56
- pycocotools==2.0.7
- pyyaml

An example of how to run the dataloader and extract samples using the example config file (_configs/WeedAI_plant.yml_) is shown in the python script _example.py_.

## Crop and Weed Dataset

To evaluate on the CAW dataset for this work we needed to generate pseudo-labels for panoptic segmentation. These labels were generated using the stem location supplied in the dataset and the connected components algorithm. These labels along with our training/validation/evaluation splits can be downloaded from [here](https://uni-bonn.sciebo.de/s/fIJHA74SMpVYgNX). Please note, these pseudo-labels are noisy as stated in the paper and are only generated for the background/crop class.

***
[1] Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena. **The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation**, _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, 2023
