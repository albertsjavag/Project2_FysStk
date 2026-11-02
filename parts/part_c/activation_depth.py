"""
Validation MSE for different activations, layers and nodes on the Runge data.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 2 - Snippet
Role: Loops by us. Claude added the validation split, the MSE grid, the printout
and the heat maps.

Verification: Ran the script and checked the printout against figures/activation_depth.png.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error

from src.neural_network import NeuralNetwork
from src.data import generate_data, split_data
from src.activations import ACTIVATIONS, identity, identity_prime
from src.costs import mse, mse_der
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
x_train, x_val, y_train, y_val = split_data(x_train, y_train)

X_train_nn = x_train.reshape(-1, 1)
X_val_nn = x_val.reshape(-1, 1)
y_train_nn = y_train.reshape(-1, 1)

layer_list = [1, 2, 3, 4]
node_list = [5, 10, 25, 50, 100]

fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.0), sharex=True)
titles = {"sigmoid": "Sigmoid", "relu": "ReLU", "leaky_relu": "Leaky ReLU"}

for name, ax in zip(["sigmoid", "relu", "leaky_relu"], axes):
    f, f_der = ACTIVATIONS[name]
    val_grid = np.zeros((len(layer_list), len(node_list)))
    for i, layers in enumerate(layer_list):
        for j, nodes in enumerate(node_list):
            hidden = [nodes] * layers 
            sizes = hidden + [1] 
            funcs = [f] * layers + [identity] 
            ders = [f_der] * layers + [identity_prime]
            net = NeuralNetwork(1, sizes, funcs, ders, mse, mse_der)
            net.train(X_train_nn, y_train_nn, method="adam", gamma=0.01, epochs=1000, batch_size=16)
            train_mse = mean_squared_error(y_train, net.predict(X_train_nn).ravel())
            val_grid[i, j] = mean_squared_error(y_val, net.predict(X_val_nn).ravel())
            print(f"{name:10s} {hidden}: train {train_mse:.4f}, val {val_grid[i, j]:.4f}")
    sns.heatmap(val_grid, ax=ax, annot=True, fmt=".4f", xticklabels=node_list, yticklabels=layer_list,
                norm=LogNorm(vmin=0.0095, vmax=0.055), annot_kws={"size": 6}, cbar=False)
    i_best, j_best = np.unravel_index(np.argmin(val_grid), val_grid.shape)
    ax.add_patch(plt.Rectangle((j_best, i_best), 1, 1, fill=False, ec="cyan", lw=1.6))
    ax.set_title(titles[name], fontsize=9)
    ax.set_ylabel("Hidden layers")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", labelrotation=0)

axes[-1].set_xlabel("Nodes per layer")
cbar = fig.colorbar(axes[0].collections[0], ax=axes, shrink=0.5, label="Validation MSE")
cbar.set_ticks([0.01, 0.02, 0.03, 0.05], labels=["0.01", "0.02", "0.03", "0.05"])
cbar.ax.minorticks_off()
save_fig(fig, "activation_depth")