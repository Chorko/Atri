import numpy as np


class NeuralNetwork:
    
    def __init__(self, layer_sizes, activation='sigmoid', weight_init='xavier', seed=42):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation
        
        np.random.seed(seed)
        
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            if weight_init == 'xavier':
                limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
                W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            else:
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        
        print(f"initialized {self.num_layers}-layer network: {layer_sizes}")
    
    def _activate(self, z):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        else:
            raise ValueError(f"unknown activation: {self.activation}")
    
    def _softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        cache = {'a': [X], 'z': []}
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            cache['z'].append(z)
            if i == self.num_layers - 1:
                a = self._softmax(z)
            else:
                a = self._activate(z)
            cache['a'].append(a)
        return a, cache
    
    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
