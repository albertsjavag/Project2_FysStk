import numpy as np

from src.data import load_fashion_mnist, split_data
from src.neural_network import NeuralNetwork
from src.activations import softmax
from src.costs import cross_entropy


X_train, y_train, X_test, y_test = load_fashion_mnist()
X_train, X_val, y_train, y_val = split_data(X_train, y_train)

T = np.eye(10)[y_train]

gammas = np.logspace(-4, -1, 4)

best_acc = 0
best_gamma = None

for gamma in gammas: 
    logreg = NeuralNetwork(784, [10], [softmax], [None], cross_entropy, None)
    history = logreg.train(X_train, T, method="adam", gamma=gamma, epochs=10, batch_size=100)
    val_acc = np.mean(np.argmax(logreg.predict(X_val), axis=1) == y_val)
    train_acc = np.mean(np.argmax(logreg.predict(X_train), axis=1) == y_train) 
    if val_acc > best_acc:
        best_acc = val_acc
        best_gamma = gamma
    print(f"gamma: {gamma:.0e}, validation accuracy {val_acc:.4f}, training accuracy {train_acc:.4f}")
    print("cost per epoch:", np.round(history, 3)) 
print(" ")
print(f"best gamma {best_gamma:.0e}, validation accuracy {best_acc:.4f}")    




