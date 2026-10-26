from sklearn.metrics import mean_squared_error

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.neural_network import NeuralNetwork
from src.activations import sigmoid, sigmoid_prime, identity, identity_prime
from src.costs import mse, mse_der

# Data, same as in project 1
x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)

# OLS with degree 12, scaled as in project 1
y_train_c, y_mean = center_y(y_train)
X_train = design_matrix(x_train, 12)
X_test = design_matrix(x_test, 12)
X_train_s, X_test_s = scale_matrix(X_train, X_test)
theta = ols(X_train_s, y_train_c)
print("OLS:", mean_squared_error(y_test, X_test_s @ theta + y_mean))

X_train_nn = x_train.reshape(-1, 1)
X_test_nn = x_test.reshape(-1, 1)
y_train_nn = y_train.reshape(-1, 1)

for hidden in [[50], [100], [50, 50], [100, 100]]:
    sizes = hidden + [1]
    funcs = [sigmoid] * len(hidden) + [identity]
    ders = [sigmoid_prime] * len(hidden) + [identity_prime]
    net = NeuralNetwork(1, sizes, funcs, ders, mse, mse_der)
    net.train(X_train_nn, y_train_nn, method="adam", gamma=0.01, epochs=1000, batch_size=16)
    print(f"NN {hidden}:", mean_squared_error(y_test, net.predict(X_test_nn).ravel()))