"""
Tests of the network against jax.grad and basic sanity checks.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 4 - Substantial
Role: Tests proposed and written by Claude.

Verification: Ran pytest; all tests pass.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)       # 64-bit, ellers avviker jax med ~1e-7
import pytest


from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, identity, identity_prime, softmax
from src.costs import mse, mse_der, cross_entropy


def make_runge_net():
    """The 1 -> 50 -> 50 -> 1 sigmoid network used in the tests."""
    return NeuralNetwork(1, [50, 50, 1],
                         [sigmoid, sigmoid, identity],
                         [sigmoid_prime, sigmoid_prime, identity_prime],
                         mse, mse_der)


def test_layer_shapes():
    net = NeuralNetwork(1, [50, 50, 1],
                        [sigmoid, sigmoid, identity],
                        [sigmoid_prime, sigmoid_prime, identity_prime],
                        mse, mse_der)
    shapes = [(W.shape, b.shape) for W, b in net.layers]
    assert shapes == [((1, 50), (50,)), ((50, 50), (50,)), ((50, 1), (1,))]


def test_predict_shape():
    net = make_runge_net()
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    assert net.predict(X).shape == (10, 1)


def test_saver_matches_predict():
    net = make_runge_net()
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    layer_inputs, zs, a = net._feed_forward_saver(X)
    assert np.allclose(a, net.predict(X))
    assert len(layer_inputs) == len(zs) == 3


def test_gradient_shapes():
    net = make_runge_net()
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    grads = net.compute_gradient(X, X ** 2)
    for (dW, db), (W, b) in zip(grads, net.layers):
        assert dW.shape == W.shape
        assert db.shape == b.shape


def test_backprop_matches_jax():
    net = make_runge_net()
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = X ** 2
    mine = net.compute_gradient(X, y)
    ref = net.jax_gradient(X, y, [jax.nn.sigmoid, jax.nn.sigmoid, lambda z: z])
    for (dW, db), (dW_j, db_j) in zip(mine, ref):
        assert np.allclose(dW, dW_j)
        assert np.allclose(db, db_j)


def test_training_lowers_cost():
    net = make_runge_net()
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 1 / (1 + 25 * X ** 2)
    history = net.train(X, y, method="adam", gamma=0.01, epochs=50, batch_size=16)
    assert history[-1] < history[0]

def test_train_rejects_1d_targets():
    net = make_runge_net()
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = X.ravel() ** 2      
    with pytest.raises(ValueError):
        net.train(X, y, epochs=1)

def test_l2_gradient():                            
    net = make_runge_net()
    net.lmbda, net.norm = 0.1, "l2"
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = X ** 2
    mine = net.compute_gradient(X, y)
    ref = net.jax_gradient(X, y, [jax.nn.sigmoid, jax.nn.sigmoid, lambda z: z])
    for (dW, db), (dW_j, db_j), (W, b) in zip(mine, ref, net.layers):
        assert np.allclose(dW, dW_j + 2 * 0.1 * W)
        assert np.allclose(db, db_j)


def test_softmax_matches_jax():
    """The shortcut delta^L = (a - y)/n for softmax + cross entropy, as in week 42 cell 59."""
    net = NeuralNetwork(4, [5, 3], [sigmoid, softmax], [sigmoid_prime, None], cross_entropy, None)
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(6, 4))
    Y = np.eye(3)[rng.integers(0, 3, size=6)]
    mine = net.compute_gradient(X, Y)
    ref = net.jax_gradient(X, Y, [jax.nn.sigmoid, jax.nn.softmax])
    for (dW, db), (dW_j, db_j) in zip(mine, ref):
        assert np.allclose(dW, dW_j)
        assert np.allclose(db, db_j)
