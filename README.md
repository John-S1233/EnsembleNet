# EnsembleNet

**EnsembleNet** is a Neural Architecture Search (NAS) framework that generates and evaluates multiple deep learning models, ultimately combining their strengths through a meta-learner. This method aims to improve model performance by leveraging diverse architectures trained on the same dataset and combining their predictions.

## Key Features
- **Neural Architecture Search**: Automatically generates and trains multiple neural network architectures (CNN, RNN, DenseNet, etc.) on CIFAR-100.
- **Meta-Learning**: Uses a meta-learner to combine the output of multiple models, boosting overall performance.
- **Multiple Architectures**: Supports a wide variety of neural architectures, including convolutional, recurrent, and hybrid models.
- **Ensemble Learning**: Aims to enhance performance by ensembling models rather than relying on a single architecture.

## Architecture Overview
The architecture of EnsembleNet consists of several components:

1. **Data Input**: The CIFAR-100 dataset is used as the primary dataset for training and evaluation.
2. **Model Generation**: Multiple models are generated based on different architecture types and hyperparameters:
   - Transformer
   - Convolutional Neural Network (CNN)
   - Densely Connected Convolutional Network (DenseNet)
   - Inception
   - MobileNet
   - Liquid Neural Network (LNN)
   - Attention Convolutional Neural Network
   - Recurrent Neural Network (RNN)
   - Hybrid Network - combines convolutional layers with recurrent layers
   - Generative Adversarial Network (GAN
   
   
4. **Meta Learner**: The meta-learner leverages a multi-branch approach that integrates sub-model predictions, encoded hyperparamates, and CIFAR-100 image data through seperate pathways to enhance overall classification performance.

Below is a diagram representing the EnsembleNet pipeline:

![Ensemble](https://github.com/user-attachments/assets/40c4cb9c-59af-4eed-b2bb-48f256192d1a)

## Dataset: CIFAR-100

![image](https://github.com/user-attachments/assets/82c13d68-fa97-4906-bfcc-3debe2f34234)

EnsembleNet is built to work with the **CIFAR-100** dataset, a collection of 60,000 32x32 color images in 100 classes, with 600 images per class. Below is an example of images from the dataset:


## How It Works

1. **Data Preprocessing**: Data is loaded and preprocessed from the CIFAR-100 dataset.
2. **Model Generation**: A pool of models with varying architectures is created. These models are trained individually on the dataset.
3. **Meta-Learning**: The meta-learner takes the predictions of all models along with the model architectures and learns to combine them for better performance.
4. **Evaluation**: The best-performing ensemble of models is selected based on validation accuracy.

## Results

EnsembleNet consistently improves model accuracy by combining the strengths of diverse
architectures. Each sub-model contributs to a better overall model through ensemble learning
which is further refined by the meta-learner. 
