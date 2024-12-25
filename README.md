# FundamentalML
This is a repository of fundamental ML projects, and derivations of them, implemented in pytorch in jupyter notebook

# MNIST Variational Autoencoder (VAE) - Kaggle Notebook

## Overview
This notebook implements a Variational Autoencoder (VAE) trained on the MNIST dataset using PyTorch Lightning. The implementation is optimized for Kaggle's notebook environment and includes interactive visualizations.

## Features
- Complete VAE implementation with encoder and decoder
- Training progress visualization
- Example reconstructions
- Random sample generation
- Simplified for Kaggle environment
- Interactive progress bars and plots

## Usage
1. Create a new Kaggle notebook
2. Copy the code into a code cell
3. Run the cell to train the VAE and see the results

## Model Architecture
- Encoder: 784 → 400 → 400 → 20 (latent space)
- Decoder: 20 → 400 → 400 → 784
- ReLU activations
- Sigmoid output layer
- Uses reparameterization trick for backpropagation

## Training Details
- Dataset: MNIST
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Binary Cross-Entropy + KL Divergence
- Epochs: 10 (adjustable)

## Visualizations
The notebook automatically generates:
1. Training progress plot (loss over time)
2. Original vs reconstructed image comparisons
3. Random samples from the trained model

## Requirements
The notebook will automatically install:
- PyTorch Lightning
- tqdm

## Notes
- Training time: ~5-10 minutes on Kaggle GPU
- Memory usage: ~2GB
- Adjust batch size if running into memory issues
- Increase epochs for better results

## Acknowledgments
Based on the original VAE paper:
"Auto-Encoding Variational Bayes" by Kingma and Welling (2013)
