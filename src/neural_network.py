import numpy as np

from src.data import SEED
from src.activations import softmax

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        lmbda=0.0,
        norm=None,
        seed=SEED
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lmbda = lmbda
        self.norm = norm
        self.rng = np.random.default_rng(seed)
        self.layers = self._create_layers()

    def _create_layers(self):
        layers = []
        i_size = self.network_input_size                     
        for layer_output_size in self.layer_output_sizes:
            scale = np.sqrt(1/i_size)           
            W = self.rng.normal(0.0, scale, (i_size, layer_output_size))
            b = np.zeros(layer_output_size) + 0.01
            layers.append((W, b))
            i_size = layer_output_size
        return layers


    def update_weights(self, layer_grads):
        pass

    def train(self, inputs, targets, method, gamma, epochs, batch_size):
        ...

    # These last two methods are not needed in the project, but they can be nice to have!
    # The first one has a layers parameter so that you can use jax.grad on it.
    def jax_compliant_predict(self, layers, inputs):
        pass

    def jax_gradient(self, inputs, targets):
        pass
