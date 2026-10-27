"""
From architecture.py: best activation sigmoid, layers 1, nodes 200, validation accuracy 0.8962
From hyperparameters.py:  best gamma 3e-03, lambda 1e-06, validation accuracy 0.8893
From logreg.py: best gamma 1e-03, validation accuracy 0.8517
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from src.neural_network import NeuralNetwork
from src.plotting import save_fig
from src.costs import cross_entropy
from src.data import load_fashion_mnist
from src.activations import sigmoid, sigmoid_prime, softmax

X_train, y_train, X_test, y_test = load_fashion_mnist()

T = np.eye(10)[y_train]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

gamma_nn = 3e-03
lmbda_nn = 1e-6
gamma_logreg = 1e-03

net = NeuralNetwork(784, [200, 10], [sigmoid, softmax], [sigmoid_prime, None], cross_entropy, None, lmbda=lmbda_nn, norm="l2")
net.train(X_train, T, method="adam", gamma=gamma_nn, epochs=10, batch_size=100)
y_pred_nn = np.argmax(net.predict(X_test), axis=1)

logreg = NeuralNetwork(784, [10], [softmax], [None], cross_entropy, None)
logreg.train(X_train, T, method="adam", gamma=gamma_logreg, epochs=10, batch_size=100)
y_pred_logreg = np.argmax(logreg.predict(X_test), axis=1)

for name, model, y_pred in [("Logistic regression", logreg, y_pred_logreg), 
                            ("Sigmoid [200]", net, y_pred_nn)]:
    acc = np.mean(y_pred == y_test) 
    n_params = sum(W.size + b.size for W, b in model.layers)
    print(f"{name}: test accuracy {acc:.4f}, parameters {n_params}")

print("Report for network:")
print(confusion_matrix(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn, target_names=class_names))

print("Report for logistic regression:")
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg, target_names=class_names))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
for ax, title, y_pred in zip(axes, ["Logistic regression", "Sigmoid [200]"],
                             [y_pred_logreg, y_pred_nn]):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true",
                                            display_labels=class_names, ax=ax,
                                            colorbar=False, values_format=".2f",
                                            xticks_rotation=45, text_kw={"fontsize": 5})
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
axes[1].set_ylabel("")
axes[1].set_yticklabels([])
save_fig(fig, "fashion_confusion")


n_show = 8
probs = net.predict(X_test)
pred = y_pred_nn 
wrong = np.where(pred != y_test)[0]
fig, axes = plt.subplots(2, n_show // 2, figsize=(3.4, 2.3))
for ax, k in zip(axes.ravel(), wrong[:n_show]):
    ax.imshow(X_test[k].reshape(28, 28), cmap="gray_r")
    ax.set_title(f"{class_names[y_test[k]]} → {class_names[pred[k]]}\n{probs[k, pred[k]]:.0%}",
                 fontsize=5.5, pad=2)
    ax.axis("off")
fig.suptitle(f"{len(wrong)} of the 10 000 test images are misclassified: "
             f"the first {n_show}", fontsize=7)

save_fig(fig, "fashion_misclassified")
