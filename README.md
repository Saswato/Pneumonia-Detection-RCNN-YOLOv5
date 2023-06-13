# Pneumonia Detection in Chest X-Rays using RCNN and YOLOv5

Saswato Bhattacharyya, Apurva Nehru, and Gaurav Thorat  
College Of Engineering, Northeastern University  
May 6, 2021

## Abstract
In this project, we aim to predict whether a given chest X-ray (CXR) image is inflicted with pneumonia or not. We employ two pretrained models, Mask-RCNN and YOLOv5, for pneumonia detection. Our dataset consists of labeled CXR images in DCM format, which are converted to JPG for YOLOv5. We compare the performance of both models and find that Mask-RCNN achieves an mAP of 98.15, while YOLOv5 achieves 57.5.

## Introduction
Accurately diagnosing pneumonia is challenging due to various factors affecting chest X-ray (CXR) interpretation. This project focuses on utilizing RCNN and YOLOv5 models for pneumonia detection in CXR images. We compare the two models based on their performance. The dataset used comprises labeled CXR images in DCM format, converted to JPG for YOLOv5. The evaluation reveals that Mask-RCNN achieves an mAP of 98.15, while YOLOv5 achieves 57.5.

## Proposed Method
To detect the presence of pneumonia in CXR images, we train a Convolutional Neural Network (CNN) using labeled images. Two models, Mask-RCNN and YOLOv5, are chosen for this task. The Mask-RCNN model is built on top of Faster RCNN, while YOLOv5 is built on top of YOLOv4. The images are converted to JPG format, and each model is trained and evaluated to determine the better performing model.

## Project Workflow
![image](https://github.com/Saswato/Pneumonia-Detection-RCNN-YOLOv5/assets/67147010/69870b12-342c-4050-83bf-97a7b7e142ad)


### Introduction to Mask-RCNN
Mask-RCNN is a Region-Based Convolutional Neural Network used for object detection. It generates region proposals and extracts feature vectors for each region. The model consists of a classification branch, a bounding box regression branch, and an additional branch for predicting object masks.

### Training with Mask-RCNN
The Mask-RCNN model is trained using the Matterport's Mask-RCNN implementation and COCO pre-trained weights. The training process involves formatting the DICOM images to JPG, annotating them, splitting the data into training and validation sets, and monitoring the train and validation loss.

### Training with YOLOv5
YOLOv5, implemented in PyTorch, is another model used for object detection. It includes different model sizes (YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x). The training process involves installing the Ultralytics implementation of YOLOv5, downloading COCO pre-trained weights, formatting DICOM images to JPG, splitting the data, training the model, and monitoring the train and validation loss.

## Results
The results show that Mask-RCNN achieves better performance compared to YOLOv5. The mAP (mean Average Precision) of Mask-RCNN is 98.15, while YOLOv5 achieves an mAP of 57.5. The training and validation performance for both models is visualized and presented.

## Conclusion
Based on the evaluation, we conclude that Mask-RCNN outperforms YOLOv5 in terms of pneumonia detection in CXR images. The project demonstrates the effectiveness of pretrained models and their potential for accurate diagnosis. Further improvements and research can be conducted to enhance the performance and

