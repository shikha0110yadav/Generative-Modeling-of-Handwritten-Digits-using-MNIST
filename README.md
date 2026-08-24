# Generative Machine Learning on MNIST

Implementation and experimental study of **generative models for handwritten digit recognition**, covering Gaussian modeling, Gaussian Mixture Models (GMMs), Expectation-Maximization (EM), generative classification, and classification with missing features.

## Overview

This project explores how probabilistic models can learn the distribution of MNIST handwritten digits and use the learned distributions for classification and reconstruction.

The MNIST images are normalized and represented as 28×28 grayscale images before applying the generative models.

### Models & Techniques

* Spherical Gaussian modeling
* Full-covariance Gaussian modeling
* Gaussian Mixture Models (GMMs)
* Expectation-Maximization (EM)
* k-means++ initialization
* Generative classification
* Missing-feature classification
* Conditional Gaussian reconstruction

## Experiments

### 1. Gaussian Generative Modeling

Compare simple Gaussian models with different covariance assumptions to understand their ability to capture variations in handwritten digits.

### 2. GMM with EM

Train Gaussian Mixture Models using **k-means++ initialization followed by EM**, allowing each digit class to be represented using multiple Gaussian components.

### 3. Generative Classification

Learn a class-conditional generative model for each MNIST digit and classify test images using the corresponding likelihoods.

### 4. Missing-Feature Classification

Artificially remove a portion of image pixels and use the learned generative distribution to:

* classify the incomplete image
* infer missing pixel values
* reconstruct the image

## Results

| Model / Experiment            |   Accuracy |
| ----------------------------- | ---------: |
| Single Gaussian per class     | **85.73%** |
| 5-component GMM               |     81.46% |
| 7-component GMM               | **83.90%** |
| 10-component GMM              |     83.00% |
| 15-component GMM              |     78.42% |
| Censored-image classification | **78.57%** |

The experiments demonstrate that increasing the number of mixture components does not necessarily improve classification performance.

## Tech Stack

* Python
* NumPy
* Matplotlib
* MNIST
* Multivariate Gaussian Models
* Gaussian Mixture Models
* Expectation-Maximization

## Project Structure

```text
Generative-ML-MNIST/
│
├── lec13.ipynb
├── mnist/
└── README.md
```

The notebook uses the `cs771` helper package for plotting and utility functions.

## Setup

### Install dependencies

```bash
pip install numpy matplotlib jupyter
```

### Dataset

Download the MNIST dataset and place it under:

```text
mnist/
```

The notebook expects the standard MNIST IDX files for training and test images/labels.

### Run

```bash
jupyter notebook lec13.ipynb
```

## Key Takeaways

* Full covariance models capture pixel correlations that spherical models cannot.
* GMMs provide a more flexible representation by modeling multiple modes within each class.
* EM provides an iterative approach for learning latent mixture assignments and Gaussian parameters.
* Generative models can also be used for inference when parts of the input are missing.

## Acknowledgements

* **MNIST** handwritten digit dataset
* `cs771` course utilities
* MNIST data-loader implementation based on the referenced dataset reader

