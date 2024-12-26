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

# DCGAN Implementation in PyTorch
This notebook implements Deep Convolutional GAN (DCGAN) following the architecture from the original [DCGAN paper](https://arxiv.org/abs/1511.06434).

## Quick Start
Run all cells in order. The notebook contains:
1. Package installation and imports
2. Model implementation (Generator and Discriminator)
3. Training loop
4. Visualization utilities

## Architecture Overview
- **Generator**: Random noise → Transposed Conv → 32x32 RGB images
- **Discriminator**: 32x32 RGB images → Strided Conv → Binary classification

## Usage
Just run the cells in order:
```python
# Training
model = train_dcgan()  # Takes about 30min on GPU

# Generate samples
plot_current_state(model)
plot_training_curves(model)
interpolate_latent_space(model)
```

## Required Packages
```python
!pip install pytorch-lightning torchvision
```

## Training Parameters
- Dataset: CIFAR-10 (downloads automatically)
- Batch size: 128
- Learning rate: 0.0002
- Epochs: 25 (adjust in training cell)

## Visualization Functions
- `plot_current_state(model)`: Grid of generated samples
- `plot_training_curves(model)`: G and D losses
- `interpolate_latent_space(model)`: Smooth transitions
- `save_samples(model, filename)`: Save high-res samples

## GPU Usage
The notebook automatically uses GPU if available via `accelerator='auto'`

## Troubleshooting
- If memory error: Reduce batch_size
- If poor results: Increase epochs
- If CUDA error: Restart runtime

# Conditional VAE vs Conditional GAN Implementation
This notebook implements and compares two conditional generative models: cVAE and cGAN.

## Models Overview

### Conditional VAE (cVAE)
- Extends VAE with class conditioning
- Both encoder and decoder see class labels
- Trained with reconstruction + KL loss
- More stable training, deterministic outputs

### Conditional GAN (cGAN)
- Extends DCGAN with class conditioning
- G and D both receive class information
- Trained adversarially
- Potentially sharper outputs, more varied samples

## Quick Start
Run all cells in order:
```python
# Train both models
cvae_model = train_cvae()
cgan_model = train_cgan()

# Compare results
compare_models(cvae_model, cgan_model)
```

## Dataset
- MNIST (10 classes, downloads automatically)
- Shows clear conditioning effects
- Easy to validate class-conditional generation

## Conditioning Methods
```python
# Generate specific digits
samples_cvae = cvae_model.generate(class_label=5)  # Generate '5's
samples_cgan = cgan_model.generate(class_label=5)  # Generate '5's

# Interpolate between classes
interpolate_classes(model, start_class=0, end_class=9)
```

## Comparison Metrics
- Sample quality per class
- Conditioning accuracy
- Generation diversity
- Training stability

## Visualization Functions
- Side-by-side generation comparison
- Class-conditional interpolation
- Training curves comparison
- Confusion matrices for generated samples

## Required Packages
```python
!pip install pytorch-lightning torchvision sklearn
```

## Training Tips
- cVAE: Watch reconstruction quality per class
- cGAN: Monitor discriminator per-class accuracy
- Both: Check for class-conditional mode collapse

## Outputs
Both models will generate:
- Class-specific samples
- Class interpolations
- Quality metrics per class
- Generated sample grids

## GPU Usage
Automatic GPU detection and usage through PyTorch Lightning
