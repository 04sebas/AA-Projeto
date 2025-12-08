import random

import numpy as np


def compute_layer_outputs(inputs, weights, biases, activation_function):
    z = np.dot(inputs, weights) + biases
    outputs = activation_function(z)
    return outputs


class NeuralNetwork:

    def __init__(self, input_size, output_size, hidden_architecture, hidden_activation, output_activation):
        self.output_weights = None
        self.output_bias = None
        self.hidden_weights = None
        self.hidden_biases = None
        self.input_size = input_size

        self.hidden_architecture = hidden_architecture

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.output_size = output_size
        self.weights = None

    def compute_num_weights(self):
        num_weights = 0
        input_size = self.input_size
        for hidden_layer_n in self.hidden_architecture:
            num_weights += (input_size+1) * hidden_layer_n
            input_size = hidden_layer_n
        num_weights += (input_size + 1) * self.output_size
        return num_weights

    def load_weights(self, weights):
        w = np.array(weights)
        self.weights = w
        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w+n])
            self.hidden_weights.append(w[start_w+n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        end_w = start_w + (input_size + 1) * self.output_size
        self.output_bias = w[start_w:start_w + self.output_size]
        self.output_weights = w[start_w + self.output_size:end_w].reshape(input_size, self.output_size)



    def forward(self, x):
        input = x
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            input = compute_layer_outputs(x, weights, bias, self.hidden_activation)

        output = compute_layer_outputs(input, self.output_weights, self.output_bias, self.output_activation)

        return output

    def forward_with_cache(self, x):
        activations = [x]
        zs = []

        a = x
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, W) + b
            zs.append(z)
            a = self.hidden_activation(z)
            activations.append(a)

        z = np.dot(a, self.output_weights) + self.output_bias
        zs.append(z)
        a = self.output_activation(z)
        activations.append(a)

        return a, activations, zs

    def compute_gradients(self, x, target):

        output, activations, zs = self.forward_with_cache(x)

        def hidden_deriv(z):
            return np.where(z > 0, 1.0, 0.01)

        def output_deriv():
            return 1

        delta = (output - target) * output_deriv()

        grad_output_W = np.outer(activations[-2], delta)
        grad_output_b = delta

        grad_hidden_W = []
        grad_hidden_b = []

        delta_l = delta

        for layer in reversed(range(len(self.hidden_weights))):
            z = zs[layer]
            a_prev = activations[layer]

            d_act = hidden_deriv(z)

            if layer == len(self.hidden_weights) - 1:
                W_next = self.output_weights
            else:
                W_next = self.hidden_weights[layer + 1]

            delta_l = np.dot(delta_l, W_next.T) * d_act

            grad_W_l = np.outer(a_prev, delta_l)
            grad_b_l = delta_l

            grad_hidden_W.insert(0, grad_W_l)
            grad_hidden_b.insert(0, grad_b_l)


        grads = []
        for Wg, bg in zip(grad_hidden_W, grad_hidden_b):
            grads.extend(bg.flatten())
            grads.extend(Wg.flatten())
        grads.extend(grad_output_b.flatten())
        grads.extend(grad_output_W.flatten())

        return np.array(grads)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.params = params
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.steps = 0

    def step(self, grads):
            self.steps += 1

            self.m = self.beta1 * self.m + (1 - self.beta1) * grads

            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads * grads)

            m_hat = self.m / (1 - self.beta1 ** self.steps)
            v_hat = self.v / (1 - self.beta2 ** self.steps)

            self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def create_network_architecture(input_size, output_size, neurons):

    def relu(x):
        return np.maximum(0, x)

    def output_fn(x):
        return x
   
    nn =  NeuralNetwork(input_size, output_size, neurons, hidden_activation=relu, output_activation=output_fn)
    num_weights = nn.compute_num_weights()
    weights = [random.uniform(-1, 1) for _ in range(num_weights)]
    nn.load_weights(weights)

    return nn