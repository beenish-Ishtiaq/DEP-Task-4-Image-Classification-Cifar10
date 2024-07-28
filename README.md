# Image Classification using CIFAR-10
## Project Overview
In this project, a Convolutional Neural Network (CNN) is being developed to classify images from the CIFAR-10 dataset into 10 predefined categories. The main steps involved data preprocessing, data augmentation, model building, training, and evaluation.
## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset includes the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
## Key Steps
1. **Data Augmentation and Preprocessing**: 
    - Applied techniques such as horizontal flip, random rotation, and random zoom to augment the training data and prevent overfitting.
    - Normalized pixel values to be between 0 and 1.

2. **Building the Convolutional Neural Network (CNN)**: 
    - Constructed a CNN model with multiple convolutional layers, max pooling layers, dropout layers, and dense layers.

3. **Training the Model**: 
    - Trained the model using the training dataset with data augmentation.

4. **Model Evaluation and Fine-tuning**: 
    - Evaluated the model using the test dataset and plotted the training/validation accuracy and loss over epochs.

5. **Handling Overfitting**: 
    - Used dropout and regularization techniques to mitigate overfitting.

## Model Architecture
The CNN model used in this project has the following architecture:
- Input layer: 32x32x3 (CIFAR-10 images)
- Data Augmentation layer
- Conv2D layer: 32 filters, (3,3) kernel, ReLU activation
- MaxPooling2D layer: (2,2) pool size
- Conv2D layer: 64 filters, (3,3) kernel, ReLU activation
- MaxPooling2D layer: (2,2) pool size
- Conv2D layer: 128 filters, (3,3) kernel, ReLU activation
- MaxPooling2D layer: (2,2) pool size
- Flatten layer
- Dense layer: 512 units, ReLU activation
- Dropout layer: 0.5 dropout rate
- Dense layer: 10 units, softmax activation

## Results
The model achieved an accuracy of 68% on the test dataset. Further tuning and experimentation can be done to improve the model performance.
