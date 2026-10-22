import jax.numpy as jnp


def mse(predict, target):
    return jnp.mean((predict - target) ** 2)


def mse_der(predict, target):
    return 2/ predict.size *(predict - target)  # had to go back and change len(predict) here to predict.size since len only takes avarage over number of rows.


def cross_entropy(predict, target):
    return -jnp.mean(jnp.sum(target*jnp.log(predict + 1e-12), axis=1))
