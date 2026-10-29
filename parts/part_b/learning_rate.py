"""
Validation MSE against learning rate for plain GD and the adaptive optimisers.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 3 - Skeleton
Role: First version by Claude, built on our nn_vs_ols.py and project 1 sgd_methods.py,
including the plot.

Verification: Ran the script and checked the MSE values and figures/learning_rate.png.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from src.data import generate_data, split_data
from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, identity, identity_prime
from src.costs import mse, mse_der
from src.plotting import save_fig

x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
x_train, x_val, y_train, y_val = split_data(x_train, y_train)

X_train_nn = x_train.reshape(-1, 1)
X_val_nn = x_val.reshape(-1, 1)
y_train_nn = y_train.reshape(-1, 1)

gammas = np.logspace(-4, 0, 9)
runs = [("plain GD", "plain", len(x_train)),
        ("RMSprop", "rmsprop", 16),
        ("Adam", "adam", 16)]

fig, ax = plt.subplots()

for label, method, batch_size in runs:
    val_mse = []
    for gamma in gammas:
        net = NeuralNetwork(1, [50, 1], [sigmoid, identity], [sigmoid_prime, identity_prime], mse, mse_der)
        net.train(X_train_nn, y_train_nn, method=method, gamma=gamma, epochs=1000, batch_size=batch_size)
        val_mse.append(mean_squared_error(y_val, net.predict(X_val_nn).ravel()))
        print(f"{label:9s} gamma = {gamma:.0e}: validation MSE {val_mse[-1]:.4f}")
    best = np.argmin(val_mse)
    print(f"{label:9s} best gamma = {gammas[best]:.0e}, validation MSE {val_mse[best]:.4f}\n")
    ax.loglog(gammas, val_mse, "o-", label=label)

ax.set_xlabel(r"Learning rate $\gamma$")
ax.set_ylabel("Validation MSE")
ax.set_ylim(top=1)
ax.legend()
save_fig(fig, "learning_rate")
