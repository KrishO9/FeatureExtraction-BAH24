# Feature Extraction from Remote Sensing High-Resolution Data using AI/ML (Ex: High Tension Towers, Windmills, Electric Substations, Brick Kilns, Farm Bunds)

### Developed an AI/ML solution to automatically identify and extract features from high-resolution satellite imagery, focusing on key structures like high tension towers, windmills, and substations. The system provides:
- **Tags**: Object identification.
- **Bounding Boxes**: Object localization.
- **Masks**: Pixel-level segmentation.
### The model is deployed via a Streamlit web application that captures user feedback, storing it in a MongoDB database for continuous improvement. This solution enhances infrastructure monitoring and land use analysis.

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
- **Deep Learning Framework**: PyTorch
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

