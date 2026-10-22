"""
This file includes all activation functions we possibly need for project 2. Might not use them
all but its the functions we have been thought in weekly exercises week 41-43.
Added dictionary for easy use.
"""

import numpy as np


def sigmoid(z):
    """The logistic function, Eq. (8.22), evaluated without overflow."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    """Eq. (8.23)."""
    return np.maximum(0.0, z)


def relu_prime(z):
    return (z > 0).astype(float)


def leaky_relu(z, a=0.01):
    """Eq. (8.24). From Section 8.13 of the lecture notes."""
    return np.where(z > 0, z, a * z)


def leaky_relu_prime(z, a=0.01):
    return np.where(z > 0, 1.0, a)


def tanh(z):
    """tanh(z) = 2 sigma(2z) - 1, Section 8.6.1: centred on zero, derivative up to 1."""
    return np.tanh(z)


def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


def identity(z):
    """Linear output for regression."""
    return z


def identity_prime(z):
    return np.ones_like(z)


def softmax(z):
    """Eq. (8.75) with the shift of Eq. (5.30): rows are samples, columns classes.

    No softmax_prime: the derivative is a Jacobian, not elementwise. Paired with
    the cross entropy the output error is simply a - y, Eq. (8.69).
    """
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_prime),
               "relu": (relu, relu_prime),
               "leaky_relu": (leaky_relu, leaky_relu_prime),
               "tanh": (tanh, tanh_prime),
               "identity": (identity, identity_prime),
               "linear": (identity, identity_prime)}