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

    def predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            a = activation_func(a @ W + b)
        return a

    def cost(self, inputs, targets):
        return self.cost_fun(self.predict(inputs), targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs, zs = [], []
        a = inputs
        for (W, b), actication_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = actication_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        layer_input, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for layer in self.layers]
        n = inputs.shape[0]
        for i in reversed(range(len(self.layers))):
            layer_inputs, z, activation_der = layer_input[i], zs[i], self.activation_ders[i]
            if i == len(self.layers) - 1:
                if self.activation_funcs[-1] is softmax:
                    dC_dz = (predict - targets) / n
                else: 
                    dC_da = self.cost_der(predict, targets)
                    dC_dz = dC_da *activation_der(z)
            else: 
                (W, b) = self.layers[i + 1]           
                dC_da = dC_dz @ W.T
                dC_dz = dC_da * activation_der(z)     

            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0)

            W_i, b_i = self.layers[i]    

            if self.norm == "l2":                                                    
                dC_dW = dC_dW + 2.0 * self.lmbda * W_i                          
            elif self.norm == "l1":                                                   
                dC_dW = dC_dW + self.lmbda * np.sign(W_i)              

            layer_grads[i] = (dC_dW, dC_db)
        return layer_grads
    

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
