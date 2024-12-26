# FundamentalML
This is a repository of fundamental ML projects, and derivations of them, implemented in pytorch in jupyter notebook

# MNIST Variational Autoencoder (VAE) Comparison - MLP vs CNN

## Overview
This notebook implements and compares two variants of Variational Autoencoders (VAE) trained on the MNIST dataset:
1. MLP-based VAE using fully connected layers
2. CNN-based VAE using convolutional layers

The comparison demonstrates the advantages of using convolutional layers for image data and provides detailed visualizations of the differences.

## Key Features
- Complete implementations of both MLP and CNN VAEs
- Side-by-side comparisons of:
  - Training progress
  - Image reconstructions
  - Random samples
  - Latent space interpolation
- Quantitative comparison metrics
- Interactive visualizations
- Detailed architecture comparison

## Model Architectures

### MLP VAE
- Encoder: 784 → 400 → 400 → 20 (latent)
- Decoder: 20 → 400 → 400 → 784
- Fully connected layers with ReLU activations
- Final Sigmoid activation

### CNN VAE
- Encoder: 
  - Conv2d layers: 1 → 32 → 64 → 64 channels
  - Kernel size: 3x3
  - Stride: 2 for downsampling
  - ReLU activations
- Latent: 7×7×64 → 20
- Decoder:
  - ConvTranspose2d layers: 64 → 32 → 1 channels
  - Upsampling with stride 2
  - Final Sigmoid activation

## Advantages of CNN Implementation
1. Parameter Efficiency
   - CNNs use weight sharing
   - Fewer parameters for similar or better performance
   - Better scaling to larger images

2. Spatial Awareness
   - Preserves local image structure
   - Better at capturing spatial patterns
   - More natural image reconstructions

3. Feature Hierarchy
   - Automatically learns hierarchical features
   - Lower layers: edges, textures
   - Higher layers: complex patterns

## Training Details
- Dataset: MNIST
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Binary Cross-Entropy + KL Divergence
- Epochs: 10

## Visualizations
The notebook generates four types of comparisons:
1. Training Progress
   - Loss curves for both models
   - Convergence speed comparison

2. Reconstruction Quality
   - Original images
   - MLP reconstructions
   - CNN reconstructions

3. Random Samples
   - Samples from MLP latent space
   - Samples from CNN latent space

4. Latent Space Interpolation
   - Smooth transitions between digits
   - Comparison of interpolation quality

## Quantitative Metrics
- Parameter count comparison
- Reconstruction MSE
- Training time
- Final loss values

## Usage
1. Create a new Kaggle notebook
2. Copy the code into a code cell
3. Run the cell to train both models and see comparisons
4. Experiment with hyperparameters to see their effects

## Requirements
The notebook automatically installs:
- PyTorch Lightning
- tqdm

## Notes
- Training time: ~15-20 minutes on Kaggle GPU
- Memory usage: ~3GB
- CNN typically achieves better reconstruction quality
- MLP might train faster due to simpler architecture

## Tips for Experimentation
1. Try different latent dimensions
2. Modify architecture depths
3. Experiment with different loss weightings
4. Test on other datasets
5. Add batch normalization layers
6. Try different activation functions

## Acknowledgments
Based on:
- "Auto-Encoding Variational Bayes" by Kingma and Welling (2013)
- "Convolutional Variational Autoencoder" implementations in PyTorch

# DCGAN Implementation

Implementation of Deep Convolutional GAN (DCGAN) following the architecture from ["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"](https://arxiv.org/abs/1511.06434) by Radford et al.

## Architecture

### Generator
- Input: Random noise vector z (100-dimensional)
- Architecture follows DCGAN paper:
  - Transposed convolutions for upsampling
  - BatchNorm in every layer except output
  - ReLU activations
  - Tanh in output layer
- Progressive upsampling: 1x1 → 4x4 → 8x8 → 16x16 → 32x32
- No fully connected layers

### Discriminator
- Input: 32x32x3 images
- Strided convolutions for downsampling
- BatchNorm in all layers except first and last
- LeakyReLU activations (slope=0.2)
- Sigmoid output
- No fully connected layers

## Implementation Details

- Framework: PyTorch Lightning
- Dataset: CIFAR-10 (32x32 RGB images)
- Batch size: 128
- Learning rate: 0.0002
- Beta1: 0.5 (Adam optimizer)
- Epochs: 25

### Key Features
- Manual optimization for better training control
- Proper weight initialization (N(0, 0.02))
- High-quality image upscaling for visualization
- Training progress monitoring
- Latent space interpolation

## Results
The model generates 32x32 RGB images and includes:
- Training loss curves for both networks
- Generated sample grids
- Latent space interpolations
- High-resolution upscaled outputs

## Usage

```python
# Training
from dcgan import train_dcgan
model = train_dcgan()

# Testing and Visualization
from test_dcgan import (
    plot_current_state,
    plot_training_curves,
    interpolate_latent_space,
    save_samples
)

# Generate and visualize samples
plot_current_state(model)
plot_training_curves(model)
interpolate_latent_space(model)
save_samples(model, 'samples.png')
```

## File Structure
- `dcgan.py`: Main implementation
- `test_dcgan.py`: Visualization and testing utilities

## Requirements
```
torch
torchvision
pytorch-lightning
matplotlib
numpy
PIL
```

## Training Tips
1. Monitor D(x) and D(G(z)) to ensure balanced training
2. Watch for mode collapse (G diversity)
3. Ensure proper image normalization (-1 to 1)
4. Use manual optimization for better control

## Acknowledgments
Based on the DCGAN paper by Radford et al. Architecture and hyperparameters follow the original paper's specifications.
