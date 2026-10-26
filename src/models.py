"""
Exact copy of our Project 1 models.py file, just added logreg. 
"""

import numpy as np
from sklearn.linear_model import Lasso

from src.neural_network import NeuralNetwork
from src.activations import softmax
from src.costs import cross_entropy



def ols(X, y):
    return np.linalg.pinv(X) @ y


def ridge(X, y, lmbda):
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lmbda * np.eye(p), X.T @ y)

def lasso_fit(X, y, lmbda):
    """Minimiser of ||y - X theta||^2 / n + lmbda ||theta||_1, Eq. (3.lassoproblem).
    scikit-learn minimises ||y - X theta||^2 / (2n) + alpha ||theta||_1: alpha = lmbda/2.
    
    got this from lecture notes chapter 3. 
    """
    return Lasso(alpha=lmbda / 2.0, fit_intercept=False,
                 max_iter=10000000, tol=1e-10).fit(X, y).coef_

logreg = NeuralNetwork(784, [10], [softmax], [None], cross_entropy, None)
