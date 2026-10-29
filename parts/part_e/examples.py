"""
Plots one example image per Fashion-MNIST class.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 1 - Debugging
Role: Written by us. Claude suggested np.where and ax.axis("off").

Verification: Checked the labels against the images in figures/fashion_examples.png.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.plotting import save_fig
from src.data import load_fashion_mnist

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


X_train, y_train, X_test, y_test = load_fashion_mnist()

fig, axes = plt.subplots(2, 5, figsize=(3.4, 1.8))
for k, ax in enumerate(axes.ravel()):
    i = np.where(y_train == k)[0][0]
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray_r")
    ax.set_title(class_names[k], fontsize=7)
    ax.axis("off")       

save_fig(fig, "fashion_examples")


