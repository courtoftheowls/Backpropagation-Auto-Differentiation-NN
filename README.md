🧠 Backpropagation and Auto-Differentiation in Neural Networks

Overview

This project implements a small neural network framework from scratch, focusing on backpropagation, auto-differentiation, and gradient-based learning. It covers both theoretical derivations and hands-on coding, culminating in training and evaluating a neural network on a synthetic regression task.

The project builds custom classes for matrix operations with automatic differentiation, allowing the neural network to compute gradients and update its weights effectively.


🚀 Features

✅ Derived gradients analytically for cross-entropy loss with logistic activation

✅ Implemented an auto-diff framework with Mat and MatOperation classes

✅ Completed core operations: ReLU, matrix multiplication, mean squared error

✅ Built a flexible Network class supporting parameterized learning

✅ Trained a multi-layer neural network on synthetic data with noisy and noiseless targets

✅ Evaluated performance using MSE and R² metrics

⚙️ Tech Stack
* Python 3
* NumPy
* Custom-built matrix auto-differentiation engine (mat.py)
* Jupyter Notebooks for experimentation (a2q5.ipynb)
* Visualization: Matplotlib (heatmaps, loss curves)


📊 Key Results
* Successfully trained a neural network to model the target function: t = sin(x₁)·cos(x₂) + x₁² − x₂²
* Achieved MSE < 0.002 and R² > 99% on noiseless data
* Demonstrated the impact of noise on generalization by comparing metrics between noisy and noiseless datasets
* Visualized smooth convergence of the loss over epochs, indicating correct gradient flows and updates
