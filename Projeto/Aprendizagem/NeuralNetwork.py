import random

import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, output_size, hidden_architecture, hidden_activation, output_activation):
        self.output_weights = None
        self.output_bias = None
        self.hidden_weights = None
        self.hidden_biases = None
        self.input_size = input_size
        
        # hidden_architecture is a tuple with the number of neurons in each hidden layer
        # e.g. (5, 2) corresponds to a neural network with 2 hidden layers in which the first has 5 neurons and the second has 2
        self.hidden_architecture = hidden_architecture
        
        # The activations are functions 
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.output_size = output_size
        self.weights = None

    def compute_num_weights(self):
        num_weights = 0
        input_size = self.input_size
        for hidden_layer_n in self.hidden_architecture:         # Itera sobre cada camada escondida,
            num_weights += (input_size+1) * hidden_layer_n       # e calcula o nº de pesos em função do nº de neuronios de cada camada
            input_size = hidden_layer_n
        num_weights += (input_size + 1) * self.output_size                          # Output layer, 4 neuronios de saída
        return num_weights

    def compute_layer_outputs(self, inputs, weights, biases, activation_function):
        z = np.dot(inputs, weights) + biases
        outputs = activation_function(z)
        return outputs

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

        # Output layer
        end_w = start_w + (input_size + 1) * self.output_size
        self.output_bias = w[start_w:start_w + self.output_size]
        self.output_weights = w[start_w + self.output_size:end_w].reshape(input_size, self.output_size)



    def forward(self, x):
        input = x
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            input = self.compute_layer_outputs(input, weights, bias, self.hidden_activation)

        output = self.compute_layer_outputs(input, self.output_weights, self.output_bias, self.output_activation)

        return output

    def forward_with_cache(self, x):
        activations = [x]  # Funçoes de ativaçao
        zs = []            # Outputs de cada camada

        a = x
        for W, b in zip(self.hidden_weights, self.hidden_biases): # Calcula e guarda os outputs e funçoes de ativaçao
            z = np.dot(a, W) + b                                  # de cada camada
            zs.append(z)
            a = self.hidden_activation(z)
            activations.append(a)

        # Camada de Output
        z = np.dot(a, self.output_weights) + self.output_bias
        zs.append(z)
        a = self.output_activation(z)
        activations.append(a)

        return a, activations, zs # Devolve o output final, a lista de funçoes de ativaçao e a lista de inputs de cada camada

    def compute_gradients(self, x, target):

        output, activations, zs = self.forward_with_cache(x)

        def hidden_deriv(z):                    #LeakyReLu
            return np.where(z > 0, 1.0, 0.01)

        def output_deriv(z):                #Derivada de x
            return 1

        delta = (output - target) * output_deriv(zs[-1]) # Erro da camada de output ( output delta)

        grad_output_W = np.outer(activations[-2], delta) # Gradiente dos pesos camada de output
        grad_output_b = delta                            # Gradiente do bias da camdad de output

        # Backpropagation para as camadas escondidas
        grad_hidden_W = []
        grad_hidden_b = []

        delta_l = delta

        # Process hidden layers backwards
        for layer in reversed(range(len(self.hidden_weights))): # Iteração sobre cada camada da NN ao contrário
            z = zs[layer]
            a_prev = activations[layer]


            d_act = hidden_deriv(z) # Derivada da funçaõ de ativaçao

            # backpropagate delta
            if layer == len(self.hidden_weights) - 1:
                W_next = self.output_weights
            else:
                W_next = self.hidden_weights[layer + 1]

            delta_l = np.dot(delta_l, W_next.T) * d_act

            grad_W_l = np.outer(a_prev, delta_l) # Gradiente dos pesos da camada atual
            grad_b_l = delta_l                   # Gradiente do bias da camada atual

            grad_hidden_W.insert(0, grad_W_l)  # Colocam-se em primeria posição para manter a ordem correta
            grad_hidden_b.insert(0, grad_b_l)


        grads = []                                          # Coloca num flat numpy vector, devido à implementação da NN
        for Wg, bg in zip(grad_hidden_W, grad_hidden_b):
            grads.extend(bg.flatten())
            grads.extend(Wg.flatten())
        grads.extend(grad_output_b.flatten())
        grads.extend(grad_output_W.flatten())

        return np.array(grads)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.params = params  # referencia aos weights da NN policy
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

            self.m = np.zeros_like(params) # Vetores first e second moment
            self.v = np.zeros_like(params) # Mesmo tamanho que weights
            self.steps = 0                 # Contador de passos

    def step(self, grads): # Grads tem que ser do mesmo tamanho que params, i.e. weights
            self.steps += 1

            # Update biased first moment estimate
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads

            # Update biased second raw moment estimate
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads * grads)

            # Compute bias-corrected versions
            m_hat = self.m / (1 - self.beta1 ** self.steps)
            v_hat = self.v / (1 - self.beta2 ** self.steps)

            # Update parameters
            self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)





def create_network_architecture(input_size, output_Size, neurons):

    def relu(x):
        return np.maximum(0, x)

    def output_fn(x):
        return x
   
    nn =  NeuralNetwork(input_size, output_Size, neurons, hidden_activation=relu, output_activation=output_fn)
    num_weights = nn.compute_num_weights()
    weights = [random.uniform(-1, 1) for _ in range(num_weights)]
    nn.load_weights(weights)
    nn.weights

    return nn