# Project 2, FYS-STK3155/FYS4155: a feed-forward neural network from scratch

**Authors:** Albert Sjåvåg and Teodor Aursnes (University of Oslo, autumn 2026)

We write our own feed-forward neural network with backpropagation and use it for
regression of the one-dimensional Runge function, comparing with OLS, Ridge and
Lasso from project 1, and for classification of Fashion-MNIST, comparing with
logistic regression and scikit-learn. The gradients are checked against `jax.grad`.

The report is in `report/Project2.pdf`.

## Installation

```bash
pip install -r requirements.txt
```

Fashion-MNIST (70 000 images, about 30 MB) is downloaded from OpenML by
scikit-learn the first time a part e script runs, and cached in `~/scikit_learn_data`.

## Structure

The code is in `src/` (shared modules) and `parts/` (one script per analysis);
there is no separate `Code` folder.

```
src/          the network and shared functions
parts/        one script per analysis, part b to e
tests/        unit tests (pytest)
figures/      all figures in the report (PNG)
outputs/      the printed output of every script
LLM_usage.md  declaration of LLM use, file by file
```

| Module | Content |
|---|---|
| `src/neural_network.py` | `NeuralNetwork`: feed-forward pass, backpropagation, L1/L2 penalty, minibatch training, and a `jax.grad` version of the gradient for testing |
| `src/activations.py` | Sigmoid, ReLU, leaky ReLU, tanh, identity and softmax, with derivatives |
| `src/costs.py` | Mean squared error and cross entropy |
| `src/optimiser.py` | Plain gradient descent, momentum, AdaGrad, RMSprop and Adam, and minibatches (from project 1) |
| `src/models.py` | OLS, Ridge and Lasso (from project 1) |
| `src/data.py` | Runge data, design matrix and scaling (from project 1), and Fashion-MNIST |
| `src/plotting.py` | Common figure settings and `save_fig` |

## Running

Every script is run from the repository root as a module, and its output is saved
in `outputs/` with the same name, for example

```bash
python -m parts.part_b.nn_vs_ols > outputs/parts.part_b.nn_vs_ols.txt
```

All random numbers use the seed 2026.

| Script | What it does | Figure |
|---|---|---|
| **Part b: regression of the Runge function** | | |
| `parts/part_b/nn_vs_ols.py` | Test MSE of OLS (degree 12) and networks with one and two hidden layers of 50 and 100 sigmoid nodes | – |
| `parts/part_b/learning_rate.py` | Validation MSE against the learning rate for plain gradient descent, RMSprop and Adam | `learning_rate.png` |
| `parts/part_b/fit.py` | OLS and a network with 5 nodes against the Runge function, and the contribution of each hidden node | `runge_fit.png` |
| **Part c: activation functions and depth** | | |
| `parts/part_c/activation_depth.py` | Validation MSE for sigmoid, ReLU and leaky ReLU against the number of layers and nodes | `activation_depth.png` |
| **Part d: L1 and L2 penalties** | | |
| `parts/part_d/norms.py` | Validation MSE against learning rate and penalty for L2 and L1, and test MSE against Ridge and Lasso from project 1 | `norms.png` |
| `parts/part_d/weights.py` | The weights of the network sorted by size, without penalty and with L2 and L1 | `weights_l1_l2.png` |
| **Part e: classification of Fashion-MNIST** | | |
| `parts/part_e/examples.py` | One training image of each class | `fashion_examples.png` |
| `parts/part_e/logreg.py` | Logistic regression (no hidden layers) for several learning rates | – |
| `parts/part_e/hyperparameters.py` | Validation accuracy against learning rate and L2 penalty, one hidden layer | `fashion_gamma_lambda.png` |
| `parts/part_e/architecture.py` | Validation accuracy for three activations against the number of layers and nodes | `Fashion_architecture.png` |
| `parts/part_e/best_model.py` | Logistic regression and the best network on the test set: accuracy, parameters, confusion matrices, precision, recall and F1, and misclassified images | `fashion_confusion.png`, `fashion_misclassified.png` |
| `parts/part_e/learning_curves.py` | Training cross entropy against epoch for logistic regression and the best network | `fashion_learning_curves.png` |
| `parts/part_e/sklearn_compare.py` | Our logistic regression and network against scikit-learn's `LogisticRegression` and `MLPClassifier` | – |

## Tests

```bash
python -m pytest -v
```

The tests check the shapes of the layers and gradients, and compare the
backpropagation gradients with `jax.grad` for mean squared error, mean squared
error with L2 penalty, and softmax with cross entropy.

## References

- H. Xiao, K. Rasul and R. Vollgraf, *Fashion-MNIST: a Novel Image Dataset for
  Benchmarking Machine Learning Algorithms*, arXiv:1708.07747 (2017).
- M. Hjorth-Jensen, lecture notes and weekly material for FYS-STK3155/FYS4155.

The use of LLMs is declared in `LLM_usage.md`.
