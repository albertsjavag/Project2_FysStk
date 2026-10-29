"""
Validation accuracy over learning rate and lambda for the Fashion-MNIST network.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 2 - Snippet
Role: Copied by us from norms.py. Claude converted it from regression to classification.

Verification: Ran the script and checked the printout against figures/fashion_gamma_lambda.png.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, softmax
from src.plotting import save_fig
from src.costs import cross_entropy
from src.data import load_fashion_mnist, split_data


X_train, y_train, X_test, y_test = load_fashion_mnist()
X_train, X_val, y_train, y_val = split_data(X_train, y_train)

T = np.eye(10)[y_train]

gammas = np.logspace(-4, -2, 5)
lmbdas = np.logspace(-6, -1, 6)

fig, ax = plt.subplots()

val_grid = np.zeros((len(gammas), len(lmbdas)))
best_acc = 0
best_gamma = None
best_lambda = None

for i, gamma in enumerate(gammas):
    for j, lmbda in enumerate(lmbdas):
        sizes = [100, 10]
        funcs = [sigmoid, softmax]
        ders = [sigmoid_prime, None]
        net = NeuralNetwork(784, sizes, funcs, ders, cross_entropy, None, lmbda=lmbda, norm="l2")
        net.train(X_train, T, method="adam", gamma=gamma, epochs=10, batch_size=100)
        train_acc = np.mean(np.argmax(net.predict(X_train), axis=1) == y_train)
        val_grid[i, j] = np.mean(np.argmax(net.predict(X_val), axis=1) == y_val)
        if val_grid[i, j] > best_acc:
            best_acc = val_grid[i, j]
            best_gamma = gamma
            best_lambda = lmbda
        print(f"l2 gamma {gamma:.0e}, lambda {lmbda:.0e}: train accuracy {train_acc:.4f}, validation accuracy {val_grid[i, j]:.4f}")
sns.heatmap(val_grid, ax=ax, annot=True, fmt=".3f", xticklabels=[f"{l:.0e}" for l in lmbdas],
            yticklabels=[f"{g:.0e}" for g in gammas])

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\gamma$")
save_fig(fig, "fashion_gamma_lambda")

print(f"best gamma {best_gamma:.0e}, lambda {best_lambda:.0e}, validation accuracy {best_acc:.4f}")