# Handwritten Digit Recognition with Multi-Stage AI Architecture

**Authors:**
- Kendall Daze-Yach
- Tobechi Nwachukwu
- Stephen Gift Akinpelu

## Abstract

This project develops a multi-staged AI architecture that demonstrates improved accuracy for handwritten digit recognition, particularly when handling varied or unexpected datapoints. The system combines two specialized models: a primary model trained on the MNIST dataset for digit recognition, and a secondary model designed to handle image variations such as flipped or blurred digits. This two-step approach showcases the AI model's ability to handle diverse situations while improving overall efficiency.

## Introduction

This project bridges the gap between traditional handwriting and digital communication by providing a solution for digitally preserving handwritten documents, letters, and forms. Our initiative offers a cost-effective storage solution for physical files while addressing the significant challenge of handwriting variability caused by factors such as noise (blurring, flipping), paper conditions, and diverse handwriting styles.

The project explores the fundamental question: **Is it better to train an AI specifically for a dataset or can we streamline using pre-processing functions?**

## Background

The project leverages the MNIST dataset as its benchmark. Created in 1994, the MNIST dataset comprises handwritten digits from high school students and employees of the United States Census Bureau. As of 2018, researchers achieved a 0.18% error rate using convolutional neural networks on the MNIST dataset, demonstrating its enduring relevance for machine learning research.

## Methodology

### Stage 1: Clean Digit Recognition
- Built an architecture to correctly identify digits in the original MNIST dataset
- Achieved an accuracy average of **99.301%**

### Stage 2: Noise Detection
- Developed an architecture to determine whether an image contains noise (flipped or blurred)
- Preprocessed MNIST dataset by randomly flipping or blurring images
- Trained the second architecture on this modified dataset

### Stage 3: Multi-Stage Integration
- Combined both architectures to handle diverse image variations
- Process flow:
  1. Second architecture determines if image is flipped or blurred
  2. Applies appropriate filter to reverse the noise
  3. Passes the corrected image to the first architecture for recognition

## Results

The multi-staged architecture demonstrated superior performance compared to standalone models:

- **Stage 1 (Clean MNIST)**: 99.301% accuracy
- **Stage 2 (Noise Detection)**: High accuracy in identifying noise type
- **Single Architecture on Noisy Data**: ~67% accuracy (significant drop)
- **Combined Multi-Stage Architecture**: **99.58% accuracy**

The results prove that multi-staged architectures significantly improve accuracy when handling diverse datasets with noise variations.

## Requirements

- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- Jupyter Notebook
- CUDA cores (highly recommended for processing)

## Installation & Setup

1. Ensure your environment contains all required libraries:
   ```bash
   pip install torch torchvision matplotlib numpy jupyter
   ```

2. Clone this repository and ensure all notebooks and `.pth` model files are in the same folder

## Usage Instructions

### Step 1: Data Preparation
**⚠️ IMPORTANT: Run this first before any other notebook**
```bash
jupyter notebook MNIST_noise_creation.ipynb
```
This notebook loads the MNIST dataset and generates the "noised dataset" required for training.

### Step 2: Main Architecture
```bash
jupyter notebook Full_Architecture.ipynb
```

**Running Instructions:**
1. This is a segmented notebook with each test outlined in the report separated for convenience
2. You do not need to run every single segment
3. **Always run all code above `# MNIST Standalone`** to properly initialize functions
4. Pre-trained `.pth` model files are provided if you want to skip to testing segments only

### Performance Optimization
- **CUDA cores are highly recommended** for processing the architecture
- If experiencing runtime issues:
  - Increase batch size (may reduce accuracy)
  - Reduce epoch count (may reduce accuracy)
  - Both changes will help with performance but may impact model accuracy

## File Structure

```
Handwritten_Digit_Recognition/
├── MNIST_noise_creation.ipynb    # Data preparation notebook
├── Full_Architecture.ipynb       # Main training and testing notebook
└── README.md                    
```

## Key Features

- **Dual Architecture System**: Separate models for noise detection and digit recognition
- **Noise Handling**: Processes flipped and blurred images effectively
- **High Accuracy**: Achieves 99.58% accuracy on diverse datasets
- **Flexible Testing**: Segmented notebook allows selective testing of components
- **Pre-trained Models**: Includes saved model files for immediate testing

This project is developed for educational and research purposes.

