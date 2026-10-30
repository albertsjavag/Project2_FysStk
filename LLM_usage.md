# LLM usage declaration

This declaration follows the course guidelines and matches the declaration in the report.

Code contribution levels: Level 0 (none), Level 1 (debugging), Level 2 (snippet),
Level 3 (skeleton), Level 4 (substantial).
All submitted code was reviewed, tested and understood by the project authors.

## Code

| File  | Level | Description |
|---|---|---|
| `src/activations.py` | 0 | Code from our week 43 exercises and Section 8.13 of the lecture notes. Found `leaky_relu` in the lecture notes |
| `src/costs.py` | 0 | Copied from our week 42 exercises (`mse` from the template, `mse_der` our own, `cross_entropy` from the lecture notebook via LLM help in week 42). |
| `src/optimiser.py` | 0 | `optimiser_step` and `make_batches` unchanged from project 1. |
| `src/neural_network.py` | 3 | Written by us from the week 42 template, with close guidance. Claude showed how our week 42 functions translate to methods (`predict`, `_feed_forward_saver`, `cost`, `compute_gradient`), combined the softmax shortcut and the L1/L2 terms in `compute_gradient`, gave a skeleton for `_create_layers`, proposed `update_weights`, `train` (from our project 1 `sgd`) and the two JAX methods, and helped find bugs. Claude added He initialisation for `relu` and `leaky_relu` in `_create_layers`, following `create_layers` in our week 43 exercises. |
| `src/models.py` | 1 | `ols`, `ridge` and `lasso_fit` from project 1. Claude found that `lasso_fit` did not converge at degree 12 and raised `max_iter` from 1e5 to 1e7. |
| `src/data.py` | 0 | Runge data from project 1. `load_fashion_mnist` is our earlier `load_mnist` found in week43 lecture notes, only changed one line to import fashion-mnist data|
| `src/plotting.py` | 0 | No contribution from Claude in this project. |
| `tests/test_implementation.py` | 4 | Test functions proposed and written by Claude (shape checks, backpropagation against `jax.grad`, training lowers the cost, rejection of 1D targets, the L2 penalty against `jax.grad`). Claude wrote `test_softmax_matches_jax`, following the softmax check in our week 42 exercises (Exercise 7, iris). |
| `parts/part_b/nn_vs_ols.py` | 2 | Data and the first network written by us. Claude found that the OLS fit lacked the scaling and centring from project 1 and added it, following our project 1 `ridge_baseline.py` (itself LLM-assisted, level 2), and wrote the loop over the four architectures. |
| `parts/part_b/learning_rate.py` | 3 | First version written by Claude, built on our `nn_vs_ols.py` and the loop structure of our project 1 `sgd_methods.py`. Claude wrote the plot of validation MSE against the learning rate. |
| `parts/part_b/fit.py` | 2 | Data, OLS and network written by us from our `nn_vs_ols.py`. Claude suggested the decomposition of the network into the terms $v_j\sigma(w_j x + b_j)$ (two lines), fixed bugs (OLS refitted on the grid, 1D targets), and wrote the plotting of the two panels. |
| `parts/part_c/activation_depth.py` | 2 | Loops over activation, layers and nodes written by us, from our `nn_vs_ols.py`. Claude added the validation split, the grid of validation MSE, the printout of training and validation MSE and the Seaborn heat maps|
| `parts/part_d/cv_norms.py` | 2 | Assembled by Claude from our week 44 exercises (the scikit-learn wrapper around our network, `GridSearchCV`, standard errors and the one-standard-error rule) and our earlier `norms.py` (heat map, comparison with Ridge and Lasso). Replaces `norms.py`, which used a single validation split. |
| `parts/part_d/weights.py` | 2 | Training loop from our `norms.py` and plotting from panel (c) of the penalty figure in the week 41 Monday lecture. Claude wrote the list of runs, the line collecting the weights of all layers, and the printout of the share of weights below 1e-3. |
| `parts/part_e/logreg.py` | 2 | Written by us from our `learning_rate.py` and week 42 exercises. Claude explained the accuracy line, suggested 10 epochs and batch size 100 (from our week 43 exercises), showed how to track the best gamma with our `norms.py` pattern, and found bugs (training twice, wrong comparison sign). |
| `parts/part_e/hyperparameters.py` | 2 | Copied by us from our `norms.py`. Claude converted the remaining regression parts to classification (network, one-hot targets, accuracy, best accuracy, heat map labels)|
| `parts/part_e/architecture.py` | 1 | Copied by us from our `activation_depth.py` and adapted to classification. Claude found bugs (missing `cost_der`, 10 instead of 784 inputs, best activation not stored, number format in the printout).|
| `parts/part_e/best_model.py` | 2 | Written by us. Claude found bugs (logistic regression given a sigmoid output layer, a missing derivative, overwritten predictions), wrote the loop printing test accuracy and number of parameters and the two-panel `ConfusionMatrixDisplay` figure, and pointed to Section 5.13 of the lecture notes for `confusion_matrix` and `classification_report`. |
| `parts/part_e/examples.py` | 1 | Written by us from the MNIST images in the week 41 Monday lecture. Claude suggested `np.where` to find one image per class and `ax.axis("off")`. |
| `parts/part_e/learning_curves.py` | 2 | Data and network from our `best_model.py`. Claude wrote the training of both models with the returned cost history and the `semilogy` plot, following the week 43 Tuesday notebook. |
| `parts/part_e/sklearn_compare.py` | 1 | Written by us from the `LogisticRegression` example in the assignment and the `MLPClassifier` example in Section 8.14 of the lecture notes. Claude found bugs (combined timing, swapped labels, training instead of test accuracy). |
