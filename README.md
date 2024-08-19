# Pneumonia Classification

## Overview

This project is the final assignment for my Deep Learning course, where I developed a method to classify X-ray images of lungs into three categories: viral pneumonia, bacterial pneumonia, and normal. The project was part of a Kaggle competition hosted by my professor, and I achieved a top 4 ranking in the class with an accuracy of 84.5%.

## Dataset

The dataset consists of X-ray images labeled into three classes: virus, bacteria, and normal. The goal was to build a model capable of accurately classifying these images into the correct category.

## Methodology

### Data Augmentation

To enhance the diversity of the training data and improve the robustness of the model, I implemented a custom data generator that applies various augmentation techniques:

- **Horizontal and Vertical Flipping**: Randomly flip images horizontally and vertically.
- **Contrast Adjustment**: Adjust the contrast of the images to simulate different lighting conditions.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Apply CLAHE to improve the contrast of the images.
- **Additive Gaussian Noise**: Introduce Gaussian noise to make the model more robust to noisy data.
- **Affine Transformations**: Apply random rotations, shear transformations, and scaling to simulate different perspectives.

### Model Architecture

I used a pretrained VGG16 model as the base for my neural network, leveraging transfer learning to achieve better performance. The architecture was modified as follows:

- **Base Model**: VGG16 pretrained on ImageNet, with the top layers removed.
- **Global Average Pooling**: Added a global average pooling layer to reduce the dimensionality of the feature maps.
- **Dense Layers**: Added fully connected layers with 1024, 512, and 128 neurons, respectively, with ReLU activation functions.
- **Output Layer**: A softmax layer with 3 neurons to classify the input images into the three classes.

### Model Training

The model was trained using a combination of the augmented data and the original dataset. I froze the majority of the VGG16 layers to retain the pretrained weights, only fine-tuning the top layers to adapt them to the new task.

### Results

- **Validation Accuracy**: 80%
- **Kaggle Competition Ranking**: Top 4 in the class with an accuracy of 84.5%.
