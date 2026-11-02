"""
5-fold cross-validation over learning rate and lambda for L1 and L2 penalties,
marking the cells within one standard error of the minimum, and a comparison with Ridge and Lasso.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, October 2026)
Level: 2 - Snippet
Role: Assembled by Claude from our week 44 exercises (scikit-learn wrapper,
GridSearchCV, standard errors, one-standard-error rule) and our earlier norms.py.

Verification: Ran the script; Ridge and Lasso test MSE match the earlier norms.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

from src.neural_network import NeuralNetwork
from src.activations import ACTIVATIONS, identity, identity_prime
from src.costs import mse, mse_der
from src.plotting import save_fig
from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ridge, lasso_fit


class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(25, 25), activation="sigmoid", gamma=0.01, lmbd=0.0, norm="l2",
                 epochs=1000, batch_size=16, seed=2026):
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.norm = norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        f, f_der = ACTIVATIONS[self.activation]
        n_hidden = len(self.hidden)
        self.net_ = NeuralNetwork(X.shape[1], [*self.hidden, 1],
                                  [f] * n_hidden + [identity],
                                  [f_der] * n_hidden + [identity_prime],
                                  mse, mse_der, lmbda=self.lmbd, norm=self.norm, seed=self.seed)
        self.net_.train(X, y.reshape(-1, 1), method="adam", gamma=self.gamma,
                        epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict(self, X):
        return self.net_.predict(X).ravel()


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
X_train, X_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1)

cv = KFold(n_splits=5, shuffle=True, random_state=2026)
gammas = [0.001, 0.003, 0.01, 0.03, 0.1]
lmbdas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.2), sharex=True)
for norm, ax, title in zip(["l2", "l1"], axes, [r"$L_2$ penalty", r"$L_1$ penalty"]):
    net = make_pipeline(StandardScaler(), NeuralNetRegressor(norm=norm))
    param_grid = {"neuralnetregressor__gamma": gammas, "neuralnetregressor__lmbd": lmbdas}
    search = GridSearchCV(net, param_grid, cv=cv, scoring="neg_mean_squared_error")
    search.fit(X_train, y_train)
    results = search.cv_results_
    cv_mse = -results["mean_test_score"].reshape(5, 5)
    se = results["std_test_score"].reshape(5, 5) / np.sqrt(cv.get_n_splits() - 1)

    i_best = np.unravel_index(np.argmin(cv_mse), cv_mse.shape)
    within = cv_mse <= cv_mse[i_best] + se[i_best]
    test_mse = mean_squared_error(y_test, search.best_estimator_.predict(X_test))
    print(f"{norm}: minimum gamma {gammas[i_best[0]]}, lambda {lmbdas[i_best[1]]}: CV MSE {cv_mse[i_best]:.4f} +/- {se[i_best]:.4f}; "
          f"test MSE {test_mse:.4f}")

    sns.heatmap(cv_mse, ax=ax, annot=True, fmt=".3f", annot_kws={"size": 6},
                xticklabels=["0", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"],
                yticklabels=[f"{g:g}" for g in gammas],
                norm=LogNorm(vmin=0.01, vmax=0.2), cbar=False)
    for i, j in zip(*np.where(within)):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, ec="white", lw=1.0))
    ax.add_patch(plt.Rectangle((i_best[1], i_best[0]), 1, 1, fill=False, ec="cyan", lw=1.6))
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(r"Learning rate $\gamma$")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", labelrotation=0)
axes[-1].set_xlabel(r"Penalty $\lambda$")
cbar = fig.colorbar(axes[0].collections[0], ax=axes, shrink=0.5, label="5-fold CV MSE")
cbar.set_ticks([0.01, 0.03, 0.1, 0.2], labels=["0.01", "0.03", "0.1", "0.2"])
cbar.ax.minorticks_off()
save_fig(fig, "cv_norms")

y_train_c, y_mean = center_y(y_train)
X_train_s, X_test_s = scale_matrix(design_matrix(x_train, 12), design_matrix(x_test, 12))
for name, model, lmb in [("Ridge", ridge, 6.9e-9), ("Lasso", lasso_fit, 2.9e-5)]:
    theta = model(X_train_s, y_train_c, lmbda=lmb)
    print(f"{name} (degree 12, lambda {lmb:g}): test MSE {mean_squared_error(y_test, X_test_s @ theta + y_mean):.4f}")
