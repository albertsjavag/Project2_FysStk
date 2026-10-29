"""
Exact copy of our Project 1 models.py file, just added logreg.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 1 - Debugging
Role: Claude found that lasso_fit did not converge and raised max_iter.

Verification: Lasso now converges at degree 12 without warnings.
"""

import numpy as np
from sklearn.linear_model import Lasso

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

