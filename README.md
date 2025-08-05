# Neural-network-from-scratch
A deep neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset. Includes manual implementation of forward propagation, backpropagation, activation functions, and training loop without any deep learning libraries.

# Handwritten Digit Classification using Deep Neural Network

This project implements a deep neural network **from scratch using only NumPy** to classify handwritten digits from the **MNIST** dataset. No deep learning libraries like TensorFlow or PyTorch are used. The goal is to understand and implement core neural network concepts such as forward propagation, backpropagation, gradient descent, and activation functions manually.

---

## 🔧 Features

- Neural network built from scratch using NumPy
- Supports multiple activation functions: **Sigmoid, ReLU, Softmax**
- Implements:
  - Forward and backward propagation
  - Mini-batch stochastic gradient descent
  - Cross-entropy loss function
  - Weight initialization and bias
  - Accuracy, precision, recall, F1-score evaluation
- Plots:
  - Training & testing loss and accuracy
  - Confusion matrix

---

## 📁 Project Structure

```bash
.
├── data_loader.py           # Loads and preprocesses MNIST dataset
├── model.py                 # Neural network implementation
├── train.py                 # Training loop
├── evaluate.py              # Model evaluation and metrics
├── utils.py                 # Helper functions (activation, loss, etc.)
├── plots/                   # Contains accuracy/loss/confusion matrix plots
├── README.md                # Project overview


🧠 How the Model Works
Input Layer: 784 (28x28 flattened pixels)

Hidden Layers: Configurable (e.g., [128, 64])

Output Layer: 10 neurons (digits 0–9) with Softmax activation

Loss: Cross-Entropy

Optimizer: Mini-Batch SGD with momentum (optional)




