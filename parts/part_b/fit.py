"""
OLS and a [5] sigmoid network fitted to the Runge function, with the five hidden-node terms.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 2 - Snippet
Role: Data, OLS and network by us. Claude suggested the decomposition into hidden-node
terms, fixed bugs and wrote the two-panel plot.

Verification: Ran the script; the sum of the terms equals the prediction (np.allclose prints True).
"""
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data, runge
from src.models import ols
from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, identity, identity_prime
from src.costs import mse, mse_der
from src.plotting import save_fig

# Data, same as in project 1
x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)

X_train_nn = x_train.reshape(-1, 1)
X_test_nn = x_test.reshape(-1, 1)
y_train_nn = y_train.reshape(-1, 1)

# OLS with degree 12, scaled as in project 1
y_train_c, y_mean = center_y(y_train)
X_train = design_matrix(x_train, 12)
X_test = design_matrix(x_test, 12)
X_train_s, X_test_s = scale_matrix(X_train, X_test)
theta = ols(X_train_s, y_train_c)
mse_ols = mean_squared_error(y_test, X_test_s @ theta + y_mean)

x_grid = np.linspace(-1, 1, 400)
X_grid = design_matrix(x_grid, 12)
_, X_grid_s = scale_matrix(X_train, X_grid)
fit_ols = X_grid_s @ theta + y_mean

net = NeuralNetwork(1, [5, 1], [sigmoid, identity], [sigmoid_prime, identity_prime], mse, mse_der)
net.train(X_train_nn, y_train_nn, method="adam", gamma=0.01, epochs=1000, batch_size=16)
mse_nn = mean_squared_error(y_test, net.predict(X_test_nn).ravel())
print(f"NN [5]: test MSE {mse_nn:.4f}")
fit_nn = net.predict(x_grid.reshape(-1, 1)).ravel()

(W1, b1), (W2, b2) = net.layers
terms = sigmoid(x_grid.reshape(-1, 1) @ W1 + b1) * W2.ravel()
print("sum of terms equals the prediction:", np.allclose(terms.sum(axis=1) + b2, fit_nn))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

axes[0].scatter(x_train, y_train, s=3, color="grey", label="data", alpha=0.3)
axes[0].plot(x_grid, fit_ols, color="blue", label="OLS, degree 12")
axes[0].plot(x_grid, fit_nn, color="red", label="NN [5]")
axes[0].plot(x_grid, runge(x_grid),"k--", color="black", label="Runge")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")
axes[0].set_title("fits")
axes[0].legend(loc="upper left", frameon=False)

for j in range(terms.shape[1]):
    axes[1].plot(x_grid, terms[:, j], lw=0.8)
axes[1].plot(x_grid, terms.sum(axis=1) + b2, "k", lw=1.8, label="sum + bias")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel(r"$v_j\,\sigma(w_j x + b_j)$")
axes[1].set_title("the five hidden nodes")
axes[1].legend()

save_fig(fig, "runge_fit")


