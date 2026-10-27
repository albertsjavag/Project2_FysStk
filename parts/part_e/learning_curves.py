import numpy as np
import matplotlib.pyplot as plt

from src.neural_network import NeuralNetwork
from src.plotting import save_fig
from src.costs import cross_entropy
from src.data import load_fashion_mnist
from src.activations import sigmoid, sigmoid_prime, softmax

X_train, y_train, X_test, y_test = load_fashion_mnist()

T = np.eye(10)[y_train]

gamma_nn = 3e-03
lmbda_nn = 1e-6
gamma_logreg = 1e-03

net = NeuralNetwork(784, [200, 10], [sigmoid, softmax], [sigmoid_prime, None], cross_entropy, None, lmbda=lmbda_nn, norm="l2")
history_nn = net.train(X_train, T, method="adam", gamma=gamma_nn, epochs=10, batch_size=100) 

logreg = NeuralNetwork(784, [10], [softmax], [None], cross_entropy, None)                              
history_logreg = logreg.train(X_train, T, method="adam", gamma=gamma_logreg, epochs=10, batch_size=100)  

epochs = np.arange(1, len(history_nn) + 1)                                                     
fig, ax = plt.subplots()                                                                              
ax.semilogy(epochs, history_logreg, "o-", color="#004488", label="Logistic regression")              
ax.semilogy(epochs, history_nn, "o-", color="#BB5566", label="Sigmoid [200]")                         
ax.set_xlabel("Epoch")
ax.set_ylabel("Training cross entropy") 
ax.legend() 
save_fig(fig, "fashion_learning_curves")