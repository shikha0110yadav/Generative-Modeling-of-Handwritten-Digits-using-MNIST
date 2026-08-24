# Generative Machine Learning on MNIST

A hands-on implementation and study of **generative models for handwritten digit data**, exploring Gaussian distributions, Gaussian Mixture Models (GMMs), Expectation-Maximization (EM), generative classification, and classification with missing image features.

## 📌 Overview

This project studies how probabilistic generative models can learn the distribution of handwritten digits and subsequently be used for both:

* **Generating new handwritten digit samples**
* **Classifying unseen MNIST images**

The project progressively builds from a simple Gaussian model to more expressive **Gaussian Mixture Models**, demonstrating how covariance structure and latent variables affect the quality of learned distributions.

The experiments use the **MNIST handwritten digit dataset**, with images represented as 28 × 28 grayscale images and flattened into 784-dimensional feature vectors.

---

## 🎯 Objectives

The project explores:

* Fitting a standard Gaussian to image data
* Learning full covariance Gaussian distributions
* Understanding the importance of covariance in generative modeling
* Learning Gaussian Mixture Models using **Expectation-Maximization**
* Using **k-means++ initialization** for GMM parameters
* Building class-conditional generative classifiers
* Comparing a single Gaussian per class against GMMs
* Studying the effect of the number of GMM components
* Performing classification when image features are missing
* Reconstructing censored images using Gaussian conditional distributions

---

## 🧠 Models Implemented

### 1. Standard Gaussian

The first model assumes:

[
X \sim \mathcal{N}(\mu, \sigma^2 I)
]

Only the mean and a scalar variance are used.

This provides a simple baseline but cannot capture correlations between different pixels.

---

### 2. Full-Covariance Gaussian

A more expressive Gaussian model is implemented as:

[
X \sim \mathcal{N}(\mu, \Sigma)
]

where both the mean vector and the complete covariance matrix are estimated using maximum likelihood.

The covariance matrix allows the model to capture dependencies between pixels and produces substantially more meaningful generated samples.

---

### 3. Gaussian Mixture Model

To model multiple modes in the data, a mixture of Gaussian distributions is implemented:

[
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x|\mu_k,\Sigma_k)
]

The GMM is trained using the **Expectation-Maximization (EM)** algorithm.

The implementation includes:

* k-means++ style centroid initialization
* Identity covariance initialization
* E-step for computing component responsibilities
* M-step for updating mixture parameters
* Mixture weights
* Multiple Gaussian components per digit class

---

## 🔬 Experiments

### Experiment 1 — Modeling Digits 0 and 4

A subset containing only handwritten **0s and 4s** is first used to study generative modeling.

The progression is:

```text
Standard Gaussian
       ↓
Full-Covariance Gaussian
       ↓
Gaussian Mixture Model
```

This experiment demonstrates how increasing the expressiveness of the probability distribution improves the ability to model variations in handwriting.

---

### Experiment 2 — Importance of Covariance

A second toy dataset is created using:

* Upright 7s
* Horizontally inverted 7s

A spherical Gaussian fails to capture the relationship between pixels.

A full covariance matrix, however, can model correlations between pixel regions and generate samples resembling either upright or inverted 7s.

This experiment highlights the importance of modeling **feature correlations** in high-dimensional generative models.

---

### Experiment 3 — Generative Classification

A separate Gaussian distribution is learned for each MNIST digit:

```text
Class 0 → Gaussian
Class 1 → Gaussian
Class 2 → Gaussian
...
Class 9 → Gaussian
```

Classification is performed using the class-conditional likelihood together with the class prior.

The implemented single-Gaussian-per-class model achieves:

**Test Accuracy: 85.73%**

---

## 🚀 GMM-Based Classification

Instead of representing each digit using a single Gaussian, a **GMM is learned independently for every digit class**.

For each class:

```text
MNIST class
    ↓
Extract class-specific samples
    ↓
k-means++ initialization
    ↓
Initialize covariance matrices
    ↓
EM optimization
    ↓
Learn K Gaussian components
    ↓
Class-conditional GMM
```

The classifier then evaluates the likelihood of a test image under every class-conditional GMM.

### GMM Component Comparison

| Components / Class | Test Accuracy |
| -----------------: | ------------: |
|                  5 |        81.46% |
|                  7 |    **83.90%** |
|                 10 |        83.00% |
|                 15 |        78.42% |

The experiments show that increasing model complexity does not necessarily improve classification performance. In this implementation, **7 components per class achieved the best test accuracy of 83.90%** among the evaluated configurations.

---

## 🧩 Classification with Missing Features

Generative models can also perform classification when portions of the input image are missing.

The experiment artificially removes the central region of MNIST images by setting approximately **21% of pixels to zero**.

The trained generative model is then used to:

1. Classify the censored image
2. Estimate missing pixel values
3. Reconstruct the image using Gaussian conditional distributions

The notebook reports a **78.57% prediction accuracy** on the censored-image experiment.

---

## 🛠️ Technologies

* **Python**
* **NumPy**
* **Matplotlib**
* **MNIST**
* Maximum Likelihood Estimation
* Multivariate Gaussian Distributions
* Gaussian Mixture Models
* Expectation-Maximization
* k-means++ Initialization
* Generative Classification
* Conditional Gaussian Reconstruction

---

## 📂 Project Structure

```text
Generative-ML-MNIST/
│
├── lec13.ipynb
├── mnist/
│   ├── train-images-idx3-ubyte/
│   ├── train-labels-idx1-ubyte/
│   ├── t10k-images-idx3-ubyte/
│   └── t10k-labels-idx1-ubyte/
│
└── README.md
```

> The notebook also uses helper modules from the `cs771` package, including plotting and utility functions.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Generative-ML-MNIST
```

### 2. Install dependencies

```bash
pip install numpy matplotlib jupyter
```

### 3. Download MNIST

Download the MNIST handwritten digit dataset and place the extracted files inside:

```text
mnist/
```

The notebook expects the standard MNIST IDX files.

### 4. Run the notebook

```bash
jupyter notebook lec13.ipynb
```

---

## 📊 Key Results

| Model                         | Test Accuracy |
| ----------------------------- | ------------: |
| Single Gaussian per class     |    **85.73%** |
| 5-component GMM per class     |        81.46% |
| 7-component GMM per class     |    **83.90%** |
| 10-component GMM per class    |        83.00% |
| 15-component GMM per class    |        78.42% |
| Censored-image classification |    **78.57%** |

---

## 💡 Key Takeaways

### Gaussian vs. Full-Covariance Gaussian

A spherical Gaussian cannot represent dependencies between pixels, while a full covariance Gaussian can capture these correlations and generate significantly better samples.

### Gaussian vs. GMM

A GMM can represent multiple modes within a class and capture different handwriting styles using multiple Gaussian components.

### Model Complexity

Increasing the number of mixture components does not automatically improve classification. The experiments demonstrate the importance of choosing an appropriate model complexity.

### Generative Models with Missing Data

Because generative models explicitly model the data distribution, they can be used not only for classification but also for reasoning about and reconstructing missing features.

---

## 📚 Concepts Demonstrated

This project provides an implementation-oriented exploration of:

* Maximum Likelihood Estimation
* Multivariate Gaussian distributions
* Covariance modeling
* Gaussian Mixture Models
* Latent-variable modeling
* Expectation-Maximization
* k-means++ initialization
* Class-conditional probability modeling
* Bayes-style generative classification
* Missing-feature inference
* Conditional Gaussian reconstruction


