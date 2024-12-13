# Waste_detection
In order to handle the project of waste detection, we are using our own Neural Network in order to detect and sort waste according to their types

# Object Detection Project from JSON Annotations
This document provides a detailed description of the project, including the model architecture, the steps implemented in the code, the data structure, as well as instructions on how to run the code and its usefulness.

## Introduction
This project aims to develop an object detection system using deep neural networks (CNN). The input data consists of images associated with JSON annotations that specify the bounding boxes and classes of the objects in each image. The model predicts the coordinates of the bounding boxes along with their corresponding classes.

## Project Architecture
## Dataset
Before training our model, we built our own dataset. 
The dataset contains 4 classes which are : "cardboard", "glass", "metal" and "plastic"

We have taken many photos of those items, sometimes individually, but mostly grouped on one image.
Due to the small size of our dataset, we tried to apply an augmentation, after labeling them.

For this, we used Roboflow, where we also grouped our data into train, test and validation.


### Model Used
- Convolutional Neural Network (CNN): The architecture is based on a convolutional network (CNN), ResNet-50, used as a pre-trained backbone. This architecture is chosen for its ability to extract robust features from images while remaining efficient in terms of computation.
- Prediction Heads:
- A classification layer that predicts the classes of the objects among 4 defined categories.
-  A regression layer that predicts the coordinates of the bounding boxes (x, y, width, height).
- Advantages of the Architecture:
- Modularity: The clear separation between feature extraction and prediction allows for easy modification or enhancement of components.
- Pre-training: The pre-trained backbone improves convergence and reduces the need for large volumes of data.
  
### Optimized Loss

Training is based on a combination of two main losses:
- Cross-Entropy Loss: Used for classifying objects into the 4 available classes.
- Smooth L1 Loss: Used for regression of the bounding box coordinates.

### Combining the Losses
The losses are combined in a way that balances the classification and object localization objectives.

```bash
alpha = 1.0 / loss_class.item()
beta = 1.0 / loss_bbox.item()
loss = alpha * loss_class + beta * loss_bbox
````

## Steps Implemented in the Code

### Loading Annotations

The annotations are loaded from a JSON file. Each annotation contains:

- The path of the associated image.
- The bounding boxes defined by their coordinates (x, y, width, height).
- The classes of the objects in the image.

### Dataset and DataLoader

- Dataset: A custom class handles the loading of images and annotations.
- Converts bounding box coordinates into a format usable by PyTorch ([xmin, ymin, xmax, ymax]).
- Applies transformations such as resizing, normalization, and conversion to tensors.
- DataLoader: Organizes the data into mini-batches and prepares it for training.
  
### Modeling

The model combines ResNet-50 for feature extraction and custom final layers for classification and regression.

### Training

- The model is trained over several epochs.
- The data is sent to the GPU if available.
- The model's weights are adjusted using the Adam optimizer.

### Visual Validation

Once the model is trained, the predicted bounding boxes are drawn on the input images to visually check their accuracy.

## Data Structure

The data is organized as follows:
```bash
├── train/
│   ├── IMG_9152.jpg
│   ├── IMG_9168.jpg
│   └── ...
└── train/_annotations.createml.json
```



## References
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#implementation
- https://blent.ai/blog/a/detection-images-yolo-tensorflow
- https://ieeexplore.ieee.org/abstract/document/9417895
- https://chatgpt.com/c/673bae07-7090-8004-8cd9-1b1180cee61b


## Project Usefulness

- Detection and Classification: Identifying objects in images with their bounding boxes and their classes (4 classes: e.g., plastic, cardboard, etc.).
- Practical Applications: This project can be adapted for waste sorting, industrial automation, or other computer vision-based use cases.
- Base for Future Improvements: The flexible architecture allows integrating new data types or models for better accuracy.
