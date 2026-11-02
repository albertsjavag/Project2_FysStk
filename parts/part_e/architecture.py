"""
Validation accuracy for different activations, layers and nodes on Fashion-MNIST.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 1 - Debugging
Role: Copied by us from activation_depth.py. Claude found bugs.

Verification: Ran the script and checked the printout against figures/Fashion_architecture.png.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.neural_network import NeuralNetwork
from src.plotting import save_fig
from src.costs import cross_entropy
from src.data import load_fashion_mnist, split_data
from src.activations import ACTIVATIONS, softmax

X_train, y_train, X_test, y_test = load_fashion_mnist()
X_train, X_val, y_train, y_val = split_data(X_train, y_train)

T = np.eye(10)[y_train]

layer_list = [1, 2, 3]
node_list = [25, 50, 100, 200]

fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True)
titles = {"sigmoid": "Sigmoid", "relu": "ReLU", "leaky_relu": "Leaky ReLU"}

best_acc = 0
best_activation = None

for name, ax in zip(["sigmoid", "relu", "leaky_relu"], axes):
    f, f_der = ACTIVATIONS[name]
    val_grid = np.zeros((len(layer_list), len(node_list)))
    for i, layers in enumerate(layer_list):
        for j, nodes in enumerate(node_list):
            hidden = [nodes] * layers 
            sizes = hidden + [10] 
            funcs = [f] * layers + [softmax] 
            ders = [f_der] * layers + [None]
            net = NeuralNetwork(784, sizes, funcs, ders, cross_entropy, None, lmbda=1e-06, norm="l2")
            net.train(X_train, T, method="adam", gamma=3e-03, epochs=10, batch_size=100)
            train_acc = np.mean(np.argmax(net.predict(X_train), axis=1) == y_train)
            val_grid[i, j] = np.mean(np.argmax(net.predict(X_val), axis=1) == y_val)
            if val_grid[i, j] > best_acc:
                best_acc = val_grid[i, j]
                best_nodes = nodes
                best_layers = layers
                best_activation = name
            print(f"{name:10s} {hidden}: train accuracy {train_acc:.4f}, val accuracy {val_grid[i, j]:.4f}")
    sns.heatmap(val_grid, ax=ax, annot=True, fmt=".3f", xticklabels=node_list, yticklabels=layer_list, vmin=0.87, vmax=0.9,
                annot_kws={"size": 6}, cbar=False)
    i_best, j_best = np.unravel_index(np.argmax(val_grid), val_grid.shape)
    ax.add_patch(plt.Rectangle((j_best, i_best), 1, 1, fill=False, ec="cyan", lw=1.6))
    ax.set_title(titles[name], fontsize=9)
    ax.set_ylabel("Hidden layers")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", labelrotation=0)

axes[-1].set_xlabel("Nodes per layer")
cbar = fig.colorbar(axes[0].collections[0], ax=axes, shrink=0.5, label="Validation accuracy")
cbar.set_ticks([0.87, 0.88, 0.89, 0.90])
save_fig(fig, "fashion_architecture")

print(f"best activation {best_activation}, layers {best_layers}, nodes {best_nodes}, "
      f"validation accuracy {best_acc:.4f}")