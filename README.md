# ALEXNET on CIFAR-10

This repository contains the implementation of the ALEXNET Convolutional Neural Network (CNN) architecture, trained and evaluated on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. ALEXNET, originally designed for the ImageNet dataset, has been adapted here to demonstrate its performance on CIFAR-10.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ALEXNET revolutionized the field of computer vision by winning the ImageNet Large Scale Visual Recognition Challenge in 2012. It introduced several key concepts in deep learning such as ReLU activation, dropout, and extensive use of GPU computation. This project adapts ALEXNET for the CIFAR-10 dataset, showcasing its versatility and robustness in handling smaller images and a different dataset.

## Architecture

The ALEXNET architecture implemented in this project consists of the following layers:

1. **Convolutional Layer 1:** 96 filters, 11x11 kernel, stride 4, ReLU activation
2. **Max-Pooling Layer 1:** 3x3 pool size, stride 2
3. **Convolutional Layer 2:** 256 filters, 5x5 kernel, stride 1, ReLU activation
4. **Max-Pooling Layer 2:** 3x3 pool size, stride 2
5. **Convolutional Layer 3:** 384 filters, 3x3 kernel, stride 1, ReLU activation
6. **Convolutional Layer 4:** 384 filters, 3x3 kernel, stride 1, ReLU activation
7. **Convolutional Layer 5:** 256 filters, 3x3 kernel, stride 1, ReLU activation
8. **Max-Pooling Layer 3:** 3x3 pool size, stride 2
9. **Fully Connected Layer 1:** 4096 neurons, ReLU activation
10. **Dropout Layer 1:** 50% dropout rate
11. **Fully Connected Layer 2:** 4096 neurons, ReLU activation
12. **Dropout Layer 2:** 50% dropout rate
13. **Fully Connected Layer 3:** 10 neurons (for 10 classes), softmax activation

## Dataset

The CIFAR-10 dataset is used for training and evaluation. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

## Prerequisites

- Python 3.7+
- TensorFlow 2.0+
- Keras
- NumPy
- Matplotlib

You can install the necessary packages using:

```sh
pip install tensorflow keras numpy matplotlib
```

## Usage

1. **Clone the repository:**

```sh
git clone https://github.com/your-username/ALEXNET-CIFAR10.git
cd ALEXNET-CIFAR10
```

2. **Train the model:**

```sh
python train.py
```

3. **Evaluate the model:**

```sh
python evaluate.py
```

4. **Visualize training history:**

```sh
python plot_history.py
```

## Results

After training, the model achieves the following performance on the CIFAR-10 test set:

- **Accuracy Score:** 61.63%

Detailed training and evaluation results, including plots of accuracy and loss over epochs, can be found in the `results` directory.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or improvements, feel free to submit an issue or a pull request.
