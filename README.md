# BreedWise\_\_Deep-Transfer-Learning-for-Dog-Classification

## Overview

This project implements deep transfer learning for accurate dog breed classification. It leverages a pre-trained Convolutional Neural Network (CNN) model fine-tuned on a dog breed dataset. The model differentiates between humans and dogs before performing classification. The goal is to achieve high accuracy while optimizing computational efficiency.

## Datasets Used

1. **Labeled Faces in the Wild (LFW) Dataset**

   - Used for human detection to ensure proper classification.
   - Download: [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

2. **Stanford Dogs Dataset**

   - Contains 120 different dog breeds with over 20,000 annotated images.
   - Download: [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

## Methodology

1. **Data Preprocessing**
   - Image resizing and normalization.
   - Data augmentation techniques such as rotation, flipping, and cropping.
   - Train-validation-test split to ensure generalization.
2. **Human Detection**
   - Implemented using HAAR cascade classifiers to differentiate between humans and dogs before classification.
3. **Transfer Learning Approach**
   - Used pre-trained CNN models (VGG16, ResNet) for feature extraction.
   - Fine-tuned the final fully connected layers for dog breed classification.
4. **Training Process**
   - Implemented **Switching Epochs Method**, which dynamically adjusts training parameters during different epochs to enhance generalization and prevent overfitting.
   - Loss function: Cross-Entropy Loss.
   - Optimizer: Adam Optimizer with an adaptive learning rate.
   - Batch size: 32.
   - Epochs: 25.
   - GPU acceleration enabled for faster training.
5. **Model Evaluation**
   - Accuracy obtained:
     - **Without Transfer Learning:**
       - Test Loss: 3.1443
       - Test Accuracy: 24.10% (2823/11715)
     - **With Transfer Learning:**
       - Test Loss: 0.5754
       - Test Accuracy: 82.46% (9660/11715)
   - Validation and training loss plots show that the model does **not overfit**, indicating a well-regularized training process.
   - Confusion matrix and precision-recall analysis conducted to evaluate misclassifications.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/dog-breed-classification.git
   cd dog-breed-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets and place them in the respective directories.
4. Run the training script:
   ```bash
   python train.py
   ```
5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results & Insights

- The **Switching Epochs Method** helped improve the model's generalization and stability.
- The **ResNet50 model with transfer learning achieved 82.46% accuracy**, significantly outperforming training without transfer learning.
- Misclassifications were reduced significantly by optimizing data preprocessing techniques.
- The training and validation loss curves show that the model is well-balanced and does not suffer from overfitting.
- Implementing transfer learning reduced training time while maintaining high performance.

## Future Improvements

- Experimenting with other pre-trained models like EfficientNet.
- Implementing real-time classification via webcam.
- Further fine-tuning to improve generalization on diverse datasets.
