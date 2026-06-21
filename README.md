# Neural-Net

A catalog of foundational deep learning concepts built from scratch and with TensorFlow/Keras, covering neural network construction, regularization, optimization algorithms, convolutional neural networks, and transfer learning. The work follows the curriculum of the Deep Learning Specialization and includes an additional notebook exploring LLM tool-use via the Anthropic API.

---

## What is Implemented

### Neural Network (from scratch with NumPy)

- Single hidden layer neural network for binary classification on a 2D planar (flower-pattern) dataset
- Forward propagation: LINEAR -> TANH -> LINEAR -> SIGMOID
- Backpropagation with cross-entropy loss
- Decision boundary visualization with matplotlib
- Comparison against logistic regression baseline

### Regularization

Three sub-topics, each in its own subdirectory:

**Weight Initialization** (`Regularization/Initializing weights/`)
- Comparison of zeros initialization, random initialization, and He initialization on a deep network
- Demonstrates vanishing/exploding gradient problems caused by poor initialization

**L2 Regularization and Dropout** (`Regularization/L2 and dropout/`)
- L2 (weight decay) regularization: cost function modification and gradient update with lambda penalty
- Dropout regularization: inverted dropout during forward pass, masking during backprop
- Applied to a 3-layer network (LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID) on a 2D dataset

**Gradient Checking** (`Regularization/GradCheck/`)
- Numerical gradient approximation using the two-sided difference formula
- Comparison of analytical gradients from backprop against numerical approximations to verify correctness

### Optimization Methods (`Optimization/`)

- Mini-batch gradient descent with configurable batch sizes
- Gradient descent with momentum (exponentially weighted moving averages of gradients)
- RMSprop (exponentially weighted moving averages of squared gradients)
- Adam optimizer (combination of momentum and RMSprop with bias correction)
- Applied to a 3-layer network (LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID) using He initialization
- Dataset: 2D classification dataset loaded from `datasets/data.mat`
- Utility: `opt_utils_v1a.py` implements forward/backward propagation, parameter initialization, cost computation, and prediction

### Convolutional Neural Networks (`CNN/`)

**From Scratch** (`CNN/From_Scratch/`)
- Convolution operation (zero-padding, single step convolution, forward pass)
- Max pooling and average pooling forward passes
- Backpropagation through convolution and pooling layers

**TensorFlow CNN Application** (`CNN/TensorFlow/`)
- CNN built with TensorFlow/Keras for image classification
- Datasets: Happy House dataset (binary: smiling/not smiling) and SIGNS dataset (6-class: digits 0-5)
- Architecture uses Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense layers
- Mini-batch training with random shuffling

**ResNet** (`CNN/Resnet.ipynb`)
- Identity block and convolutional block implementations
- Full ResNet-50 architecture assembled in Keras
- Trained on the SIGNS dataset (64x64 RGB images, 6 classes)

**Transfer Learning with MobileNetV2** (`CNN/TransferLearning/`)
- MobileNetV2 pretrained on ImageNet used as a fixed feature extractor
- Fine-tuning the last layers for a custom classification task
- Data augmentation applied during training

### TensorFlow (`Tensorflow/`)

**Basics** (`Tensorflow/Basics/`)
- TensorFlow 2 fundamentals: tensors, variables, GradientTape, `@tf.function`
- Building and training a fully connected neural network using the Sequential and Functional APIs
- Dataset: SIGNS dataset (hand sign images for digits 0-5, 64x64x3 input, 6-class softmax output)
- Architecture: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX (3 dense layers)
- One-hot encoding of labels, Adam optimizer, mini-batch training

**Intermediate** (`Tensorflow/Intermidiate/`)
- Cat vs. dog binary image classifier built with Keras
- Convolutional network with data loading and preprocessing

### Functions / LLM Tool Use (`Functions.ipynb`)

- Demonstrates function calling (tool use) with the Anthropic Claude API (`claude-3-opus-20240229`)
- Constructs a custom system prompt to instruct the model to invoke a `dollar_price` tool
- Parses the model's JSON function-call response, executes the local Python function, and injects the result back into the conversation
- Full multi-turn tool-use loop: user query -> model invokes tool -> result injected -> model produces final answer

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| From-scratch neural nets | NumPy, SciPy, matplotlib |
| Deep learning framework | TensorFlow 2 / Keras |
| Data utilities | h5py, scipy.io |
| Dataset generation | scikit-learn (`make_moons`, `make_circles`) |
| LLM API | Anthropic Python SDK (`anthropic`) |
| Notebooks | Jupyter Notebook |
| Language | Python 3 |

---

## Installation and Setup

**Prerequisites:** Python 3.8+ and pip.

```bash
git clone https://github.com/YashNagraj75/Neural-Net.git
cd Neural-Net
pip install numpy matplotlib scipy h5py scikit-learn tensorflow anthropic
```

For the Anthropic API notebook (`Functions.ipynb`), set your API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

If running on Google Colab, the notebook reads the key from Colab user secrets (`userdata.get('ANTHROPIC_API_KEY')`).

---

## Usage

All experiments are organized as Jupyter notebooks. Launch Jupyter from the repo root:

```bash
jupyter notebook
```

Then open any of the notebooks listed below.

### Notebook Guide

| Notebook | Directory | What it demonstrates |
|---|---|---|
| `Planar_data_classification_with_one_hidden_layer.ipynb` | `Neural_Network/` | Single hidden-layer NN, from-scratch backprop, planar dataset |
| `Initialization.ipynb` | `Regularization/Initializing weights/` | Zeros vs. random vs. He initialization |
| `Regularization.ipynb` | `Regularization/L2 and dropout/` | L2 regularization and dropout |
| `Gradient_Checking.ipynb` | `Regularization/GradCheck/` | Numerical gradient verification |
| `Optimization_methods.ipynb` | `Optimization/` | Mini-batch GD, Momentum, RMSprop, Adam |
| `Tensorflow_introduction.ipynb` | `Tensorflow/Basics/` | TF2 fundamentals, SIGNS 6-class classifier |
| `cat-dog-classifier.ipynb` | `Tensorflow/Intermidiate/` | Cat vs. dog CNN with Keras |
| `Convolution_model_Application.ipynb` | `CNN/TensorFlow/` | TF CNN on Happy House / SIGNS datasets |
| `Resnet.ipynb` | `CNN/` | ResNet-50 identity and convolutional blocks |
| `Transfer_learning_with_MobileNet_v1.ipynb` | `CNN/TransferLearning/` | MobileNetV2 transfer learning |
| `Functions.ipynb` | root | Anthropic API tool-use / function calling |

### Running a specific notebook

```bash
# Example: run the optimization notebook
cd Optimization
jupyter notebook Optimization_methods.ipynb
```

Datasets are included in the `datasets/` subdirectory within each module folder (HDF5 `.h5` files and `.mat` files).

---

## Expected Results

| Experiment | Expected outcome |
|---|---|
| Neural_Network planar dataset | ~90% training accuracy with a 4-unit hidden layer vs ~47% for logistic regression |
| He initialization | Faster convergence and stable training compared to zeros or random init |
| L2 / Dropout regularization | Reduced overfitting on the training set, improved generalization |
| Adam optimizer | Faster convergence than vanilla gradient descent or momentum alone |
| TF SIGNS classifier | ~80%+ test accuracy on 6-class sign digit recognition |
| ResNet-50 on SIGNS | High accuracy due to deep residual connections |
| MobileNetV2 transfer learning | Strong accuracy with minimal training thanks to ImageNet pretraining |

---

## Project Structure

```
Neural-Net/
├── Neural_Network/
│   ├── Planar_data_classification_with_one_hidden_layer.ipynb
│   ├── planar_utils.py          # Dataset loader, sigmoid, decision boundary plotter
│   ├── testCases_v2.py
│   └── public_tests.py
├── Regularization/
│   ├── Initializing weights/
│   │   ├── Initialization.ipynb
│   │   └── init_utils.py
│   ├── L2 and dropout/
│   │   ├── Regularization.ipynb
│   │   └── reg_utils.py         # Forward/backward prop, update_parameters, cost
│   └── GradCheck/
│       ├── Gradient_Checking.ipynb
│       └── gc_utils.py
├── Optimization/
│   ├── Optimization_methods.ipynb
│   ├── opt_utils_v1a.py         # 3-layer net, He init, forward/backward prop, Adam utils
│   └── datasets/
├── CNN/
│   ├── From_Scratch/            # NumPy conv and pool layers
│   ├── TensorFlow/
│   │   ├── Convolution_model_Application.ipynb
│   │   └── cnn_utils.py         # Happy House / SIGNS data loaders, mini-batches
│   ├── TransferLearning/
│   │   └── Transfer_learning_with_MobileNet_v1.ipynb
│   └── Resnet.ipynb
├── Tensorflow/
│   ├── Basics/
│   │   ├── Tensorflow_introduction.ipynb
│   │   ├── tf_utils.py          # SIGNS loader, mini-batches, one-hot encoding
│   │   └── improv_utils.py
│   └── Intermidiate/
│       └── cat-dog-classifier.ipynb
└── Functions.ipynb              # Anthropic API tool-use demo
```
