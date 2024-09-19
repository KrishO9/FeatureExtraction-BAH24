
# Satellite Image Analysis with Object-Aware Refinement in Multi-Task Deep Learning Architecture

## Overview
This project enhances satellite image analysis by incorporating an **Object-Aware Refinement Module** into a multi-task deep learning architecture. The solution integrates segmentation, object detection, and multi-label classification, using a shared backbone architecture based on **ResNet50**. The pipeline enables efficient feature extraction, segmentation, object detection, and classification of satellite images.

## Key Components
1. **ResNet50 Backbone**:
   - **Purpose**: Shared feature extractor for all tasks.
   - **Pretrained**: Yes, on ImageNet.
   - **Fine-tuning**: Initially frozen, then fine-tuned with a learning rate of 1e-5.

2. **U-Net Decoder**:
   - **Purpose**: Generates segmentation masks.
   - **Loss Function**: Combination of Dice Loss and Focal Loss.
   - **Output Activation**: Softmax/ReLU.

3. **YOLO Head**:
   - **Purpose**: Object detection, outputting bounding boxes, object classes, and confidence scores.
   - **Loss Function**: Combination of Binary Cross-Entropy, Mean Squared Error, and Categorical Cross-Entropy.
   - **Output Activation**: Sigmoid (objectness and class probabilities) and Linear (bounding box coordinates).

4. **Classification Branch**:
   - **Purpose**: Predicts image-level labels.
   - **Architecture**: Global Average Pooling followed by fully connected layers (Dropout and Batch Normalization applied).
   - **Loss Function**: Binary Cross-Entropy.
   - **Output Activation**: Sigmoid.

## Preprocessing Tools
- **QGIS (v3.28)**: For labeling satellite images (tags, bounding boxes, masks).
- **GDAL (v3.5.0)**: Essential for geospatial data format handling and satellite imagery preprocessing.
- **OpenCV**: Used for training data preparation (applying masks, segmentation, etc.).

## Model Development and Training
- **Programming Language**: Python (v3.11).
- **Deep Learning Framework**: TensorFlow (v2.11).
- **Segmentation Architecture**: U-Net.
- **Object Detection**: YOLO v8.
- **Shared Backbone**: ResNet50 (pretrained and fine-tuned).

### Training Details
- **Data Augmentation**: Random flips, rotations, color jittering, and random cropping.
- **Regularization**:
  - Weight Decay: 1e-4.
  - Dropout: 0.5 in fully connected layers.

## Web Application Deployment
- **Framework**: Streamlit (v1.17).
- **Purpose**: Interactive web app for model deployment, displaying results, and collecting user feedback.

## Beta Features 
- Tracking of similar features located within a particular region.
- Exploration of additional data augmentation techniques.
- Real-time satellite image analysis for geospatial applications.

