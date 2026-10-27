import numpy as np
import matplotlib.pyplot as plt

from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, identity, identity_prime
from src.costs import mse, mse_der
from src.data import generate_data, split_data
from src.plotting import save_fig

x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)

sizes = [25, 25] + [1]
funcs = [sigmoid, sigmoid, identity]
ders = [sigmoid_prime, sigmoid_prime, identity_prime]

runs = [("no penalty", None, 0.0, "#004488"),                           
        ("L2", "l2", 1e-5, "#BB5566"), 
        ("L1", "l1", 1e-5, "#DDAA33")] 

fig, ax = plt.subplots()
for label, norm, lmbda, color in runs:
    net = NeuralNetwork(1, sizes, funcs, ders, mse, mse_der, lmbda=lmbda, norm=norm)
    net.train(x_train.reshape(-1, 1), y_train.reshape(-1, 1),
              method="adam", gamma=1e-2, epochs=1000, batch_size=16)
    W = np.abs(np.concatenate([W.ravel() for W, b in net.layers]))
    print(f"{label}: share of |w| < 1e-3: {np.mean(W < 1e-3):.2f}")
    ax.semilogy(np.sort(W)[::-1] + 1e-12, color=color, label=label)

ax.set_xlabel("Weight, sorted by size")
ax.set_ylabel(r"$|w|$")
ax.set_ylim(1e-6, 1e1)
ax.legend()
save_fig(fig, "weights_l1_l2")