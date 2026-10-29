"""
Validation MSE over learning rate and lambda for L1 and L2 penalties, compared with Ridge and Lasso.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 1 - Debugging
Role: Written by us from a recipe from Claude. Claude fixed bugs and suggested LogNorm
for the heat map.

Verification: Ran the script; Ridge and Lasso MSE match project 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm


from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, identity, identity_prime
from src.costs import mse, mse_der
from src.plotting import save_fig
from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ridge, lasso_fit


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
x_tr, x_val, y_tr, y_val = split_data(x_train, y_train)

X_train_nn = x_tr.reshape(-1, 1)
X_val_nn = x_val.reshape(-1, 1)
y_train_nn = y_tr.reshape(-1, 1)

gammas = np.logspace(-3, -1, 5)
lmbdas = np.logspace(-8, 0, 9)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)

lmbda_ridge = 6.9e-9
lmbda_lasso = 2.9e-5

y_train_c, y_mean = center_y(y_train)
X_train = design_matrix(x_train, 12)
X_test = design_matrix(x_test, 12)
X_train_s, X_test_s = scale_matrix(X_train, X_test)

for name, model, lmb in zip(("ridge", "lasso"), (ridge, lasso_fit), (lmbda_ridge, lmbda_lasso)):
    theta = model(X_train_s, y_train_c, lmbda=lmb)
    print(f"{name}:", mean_squared_error(y_test, X_test_s @ theta + y_mean))

best = {}

for norm, ax in zip(["l2", "l1"], axes):
    val_grid = np.zeros((len(gammas), len(lmbdas)))
    best_val = np.inf
    best_gamma = None
    best_lambda = None

    for i, gamma in enumerate(gammas):
        for j, lmbda in enumerate(lmbdas):
            sizes = [25, 25] + [1]
            funcs = [sigmoid, sigmoid, identity]
            ders = [sigmoid_prime, sigmoid_prime, identity_prime]
            net = NeuralNetwork(1, sizes, funcs, ders, mse, mse_der, lmbda=lmbda, norm=norm)
            net.train(X_train_nn, y_train_nn, method="adam", gamma=gamma, epochs=1000, batch_size=16)
            train_mse = mean_squared_error(y_tr, net.predict(X_train_nn).ravel())
            val_grid[i, j] = mean_squared_error(y_val, net.predict(X_val_nn).ravel())
            if val_grid[i, j] < best_val:
                best_val = val_grid[i, j]
                best_gamma = gamma
                best_lambda = lmbda
            print(f"{norm:10s} gamma {gamma:.0e}, lambda {lmbda:.0e}: train {train_mse:.4f}, val {val_grid[i, j]:.4f}")
    sns.heatmap(val_grid, ax=ax, annot=True, xticklabels=[f"{l:.0e}" for l in lmbdas],
                yticklabels=[f"{g:.0e}" for g in gammas], norm=LogNorm(vmin=0.01, vmax=0.2),
                annot_kws={"size": 5.5}, cbar=(ax is axes[-1]))
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_title(norm)
    ax.set_xlabel(r"$\lambda$")
    best[norm] = (best_gamma, best_lambda)

axes[0].set_ylabel(r"$\gamma$")
save_fig(fig, "norms")

for norm, (gamma, lmbda) in best.items():
    net = NeuralNetwork(1, sizes, funcs, ders, mse, mse_der, lmbda=lmbda, norm=norm)
    net.train(x_train.reshape(-1, 1), y_train.reshape(-1, 1),
              method="adam", gamma=gamma, epochs=1000, batch_size=16)
    test_mse = mean_squared_error(
        y_test, net.predict(x_test.reshape(-1, 1)).ravel()
        )
    print(f"NN {norm} (gamma {gamma:.0e}, lambda {lmbda:.0e}): {test_mse}")