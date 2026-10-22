import numpy as np
from sklearn.model_selection import train_test_split
import os

SEED = 2026


def runge(x):
    return 1 / (1 + 25 * x**2)


def generate_data(n, sigma=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y_true = runge(x)
    y = y_true + rng.normal(0, sigma, n)
    return x, y, y_true


def design_matrix(x, degree):
    """Polynomial design matrix without intercept: columns x, x², ..., x^degree."""
    return np.column_stack([x**k for k in range(1, degree + 1)])


def scale_matrix(X_train, X_test=None):
    """Standardize all columns using training statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_s = (X_train - mean) / std

    if X_test is None:
        return X_train_s

    return X_train_s, (X_test - mean) / std


def center_y(y_train):
    """Center y using training mean. Returns (y_centered, mean)."""
    mean = y_train.mean()
    return y_train - mean, mean


def split_data(x, y, test_size=0.2, seed=SEED):
    return train_test_split(x, y, test_size=test_size, random_state=seed)

""" ALL above is runge data from project 1."""


def load_mnist():
    """The 70 000 MNIST images as (X_train, y_train, X_test, y_test), pixels scaled to [0, 1], 784 columns.

    Looks for the repository's mnist.npz first (no download needed) and falls
    back on scikit-learn's fetch_openml, which downloads 70 000 images once.
    """
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [os.path.join(here, "..", "mnist.npz"),
              os.path.join(here, "mnist.npz")]
    for path in candidates:
        if os.path.exists(path):
            d = np.load(path)
            X_train = d["x_train"].reshape(len(d["x_train"]), -1).astype(float) / 255.0
            X_test = d["x_test"].reshape(len(d["x_test"]), -1).astype(float) / 255.0
            return X_train, d["y_train"].astype(int), X_test, d["y_test"].astype(int)
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)
    return X[:60000], y[:60000], X[60000:], y[60000:]