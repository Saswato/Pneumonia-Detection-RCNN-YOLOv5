# Pneumonia Detection in Chest X-Rays using RCNN and YOLOv5

![xray](https://github.com/Saswato/Pneumonia-Detection-RCNN-YOLOv5/assets/67147010/7387f184-2d74-4ddf-9f01-c1a370ab4fc3)




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
![image](https://github.com/Saswato/Pneumonia-Detection-RCNN-YOLOv5/assets/67147010/9f3fc2b4-aa2b-4cf8-a0c7-aa22fd834aff)

Mask-RCNN is a Region-Based Convolutional Neural Network used for object detection. It generates region proposals and extracts feature vectors for each region. The model consists of a classification branch, a bounding box regression branch, and an additional branch for predicting object masks.


### Introduction to YOLOv5
![image](https://github.com/Saswato/Pneumonia-Detection-RCNN-YOLOv5/assets/67147010/081ae043-7c96-485b-a3e4-14e3fe47835f)

YOLO (You Only Look Once) is a real-time object detection system. YOLOv5, implemented in PyTorch, is the latest iteration of the YOLO series. It adopts a different approach compared to Mask-RCNN by dividing the input image into a grid and predicting bounding boxes and class probabilities directly. YOLOv5 includes different model sizes (YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x) that offer a trade-off between speed and accuracy.

We compare the performance of both Mask-RCNN and YOLOv5 models in pneumonia detection using our labeled CXR dataset. The images are preprocessed and formatted for training with each model.

### Training with Mask-RCNN
The Mask-RCNN model is trained using the Matterport's Mask-RCNN implementation and COCO pre-trained weights. The training process involves formatting the DICOM images to JPG, annotating them, splitting the data into training and validation sets, and monitoring the train and validation loss.

### Training with YOLOv5
YOLOv5, implemented in PyTorch, is another model used for object detection. It includes different model sizes (YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x). The training process involves installing the Ultralytics implementation of YOLOv5, downloading COCO pre-trained weights, formatting DICOM images to JPG, splitting the data, training the model, and monitoring the train and validation loss.

## Results
The results show that Mask-RCNN achieves better performance compared to YOLOv5. The mAP (mean Average Precision) of Mask-RCNN is 98.15, while YOLOv5 achieves an mAP of 57.5. The training and validation performance for both models is visualized and presented.

## Conclusion
In this project, we explored the application of Mask-RCNN and YOLOv5 models for pneumonia detection in chest X-ray images. The results indicate that Mask-RCNN outperforms YOLOv5 in terms of accuracy, achieving an mAP of 98.15. This project provides insights into the use of deep learning models for medical image analysis and opens avenues for further research to improve pneumonia detection algorithms.

## References
1. Schweitzer, D., & Agrawal, R. (2018). Multi-Class Object Detection from Aerial Images Using Mask R-CNN. 2018 IEEE International Conference on Big Data (Big Data). [DOI:10.1109/bigdata.2018.8622536](https://doi.org/10.1109/bigdata.2018.8622536)
2. [Armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector)
3. [YOLOv3-RSNA Starting Notebook | Kaggle](https://www.kaggle.com/seohyeondeok/yolov3-rsna-starting-notebook)

