"""
Compares our models with scikit-learn on test accuracy and time.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 1 - Debugging
Role: Written by us. Claude found bugs.

Verification: Ran the script; our accuracies match best_model.py.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

from src.neural_network import NeuralNetwork
from src.costs import cross_entropy
from src.data import load_fashion_mnist
from src.activations import sigmoid, sigmoid_prime, softmax


X_train, y_train, X_test, y_test = load_fashion_mnist()

T = np.eye(10)[y_train]

gamma_nn = 3e-03
lmbda_nn = 1e-6
gamma_logreg = 1e-03

# === OWN implementation logreg ===
start = time.perf_counter()

logreg = NeuralNetwork(784, [10], [softmax], [None], cross_entropy, None)
logreg.train(X_train, T, method="adam", gamma=gamma_logreg, epochs=10, batch_size=100)
acc_own_logreg = np.mean(np.argmax(logreg.predict(X_test), axis=1) == y_test)

elapsed = time.perf_counter() - start
print(f"Own logreg:     test accuracy {acc_own_logreg:.4f}, time {elapsed:.4f} s")

# === OWN implementation network ===

start = time.perf_counter()

net = NeuralNetwork(784, [200, 10], [sigmoid, softmax], [sigmoid_prime, None], cross_entropy, None, lmbda=lmbda_nn, norm="l2")
net.train(X_train, T, method="adam", gamma=gamma_nn, epochs=10, batch_size=100)
acc_own_network = np.mean(np.argmax(net.predict(X_test), axis=1) == y_test)

elapsed = time.perf_counter() - start
print(f"Own sigmoid [200]:     test accuracy {acc_own_network:.4f}, time {elapsed:.4f} s")


# === scikit-learn's implementation logreg ===
start = time.perf_counter()

model = LogisticRegression(solver='saga', max_iter=1000, random_state=2026)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred)

elapsed = time.perf_counter() - start
print(f"sklearn logreg:     test accuracy {accuracy_log:.4f}, time {elapsed:.4f} s")

# === scikit-learn's implementation MLPClassifier ===
start = time.perf_counter()
mlp = MLPClassifier(hidden_layer_sizes=(200,), activation="logistic", solver="adam",
                    learning_rate_init=3e-3, alpha=1e-6, batch_size=100,
                    max_iter=10, random_state=2026)
mlp.fit(X_train, y_train)
accuracy_MLP = mlp.score(X_test, y_test)

elapsed = time.perf_counter() - start
print(f"sklearn MLPClassifier:     test accuracy {accuracy_MLP:.4f}, time {elapsed:.4f} s")


