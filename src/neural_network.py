import numpy as np
import jax

from src.data import SEED
from src.activations import softmax, relu, leaky_relu
from src.optimiser import optimiser_step, make_batches

class NeuralNetwork:
    """Feed-forward network trained by backpropagation, with optional L1/L2 penalty.

    LLM-assisted:
    (Claude, Opus 5.5, October 2026), level 3 
    written by us from the week 42 template. Claude gave a skeleton 
    for _create_layers, proposed update_weights, train and the JAX methods, 
    and helped find bugs.

    Verified by tests/test_implementation.py: gradients agree with jax.grad for MSE,
    MSE with L2, and softmax with cross entropy. Fashion-MNIST accuracy is within
    one percentage point of scikit-learn's MLPClassifier (sklearn_compare.py).
    """

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
        for layer_output_size, activation_func in zip(self.layer_output_sizes, self.activation_funcs):
            if activation_func in (relu, leaky_relu):
                scale = np.sqrt(2/i_size)
            else:
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
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for layer in self.layers]
        n = inputs.shape[0]
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]
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
    

    def update_weights(self, layer_grads, method="plain", gamma=0.01, t=1):
        new_layers = []
        for (W, b), (dW, db), (state_W, state_b) in zip(self.layers, layer_grads, self.opt_states):
            W, state_W = optimiser_step(method, W, dW, state_W, t, gamma)
            b, state_b = optimiser_step(method, b, db, state_b, t, gamma)

            new_layers.append((W, b))
        self.layers = new_layers


    def train(self, inputs, targets, method="plain", gamma=0.01, epochs=100, batch_size=32):
        self.opt_states = [({}, {}) for _ in self.layers]
        n = inputs.shape[0]
        t = 0
        history = []
        for epoch in range(epochs):
            for batch in make_batches(n, batch_size, self.rng):
                t += 1
                layer_grads = self.compute_gradient(inputs[batch], targets[batch])
                self.update_weights(layer_grads, method, gamma, t)        
            history.append(float(self.cost(inputs, targets)))
        return history


    def jax_compliant_predict(self, layers, inputs, activation_funcs):
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            a = activation_func(a @ W + b)
        return a
    
    def jax_gradient(self, inputs, targets, jax_activation_funcs):
        def cost_of_layers(layers):
            predict = self.jax_compliant_predict(layers, inputs, jax_activation_funcs)
            return self.cost_fun(predict, targets)
        return jax.grad(cost_of_layers)(self.layers)
