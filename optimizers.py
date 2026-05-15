import numpy as np

# optimizers for the neural network
# started with SGD and momentum, will add more later

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, weights, biases, grads_w, grads_b):
        for i in range(len(weights)):
            weights[i] -= self.lr * grads_w[i]
            biases[i] -= self.lr * grads_b[i]


class MomentumGD:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_w = None
        self.v_b = None
    
    def update(self, weights, biases, grads_w, grads_b):
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            self.v_w[i] = self.beta * self.v_w[i] + grads_w[i]
            self.v_b[i] = self.beta * self.v_b[i] + grads_b[i]
            weights[i] -= self.lr * self.v_w[i]
            biases[i] -= self.lr * self.v_b[i]


def get_optimizer(name, lr=0.001):
    opts = {
        'sgd': SGD, 'momentum': MomentumGD,
    }
    if name.lower() not in opts:
        raise ValueError(f"unknown optimizer: {name}. available: {list(opts.keys())}")
    return opts[name.lower()](lr=lr)
