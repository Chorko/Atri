import numpy as np

# all the optimizers we need for the assignment
# each one has an update() method that modifies weights in-place

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


class NesterovGD:
    # nesterov accelerated gradient — look ahead then correct
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
            weights[i] -= self.lr * (self.beta * self.v_w[i] + grads_w[i])
            biases[i] -= self.lr * (self.beta * self.v_b[i] + grads_b[i])


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache_w = None
        self.cache_b = None
    
    def update(self, weights, biases, grads_w, grads_b):
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in weights]
            self.cache_b = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            self.cache_w[i] = self.beta * self.cache_w[i] + (1-self.beta) * grads_w[i]**2
            self.cache_b[i] = self.beta * self.cache_b[i] + (1-self.beta) * grads_b[i]**2
            weights[i] -= self.lr * grads_w[i] / (np.sqrt(self.cache_w[i]) + self.eps)
            biases[i] -= self.lr * grads_b[i] / (np.sqrt(self.cache_b[i]) + self.eps)


class Adam:
    # the one that usually just works lol
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = None; self.v_w = None
        self.m_b = None; self.v_b = None
        self.t = 0
    
    def update(self, weights, biases, grads_w, grads_b):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1
        
        for i in range(len(weights)):
            # first moment
            self.m_w[i] = self.beta1 * self.m_w[i] + (1-self.beta1) * grads_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1-self.beta1) * grads_b[i]
            # second moment
            self.v_w[i] = self.beta2 * self.v_w[i] + (1-self.beta2) * grads_w[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1-self.beta2) * grads_b[i]**2
            
            # bias correction
            mw_hat = self.m_w[i] / (1 - self.beta1**self.t)
            mb_hat = self.m_b[i] / (1 - self.beta1**self.t)
            vw_hat = self.v_w[i] / (1 - self.beta2**self.t)
            vb_hat = self.v_b[i] / (1 - self.beta2**self.t)
            
            weights[i] -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
            biases[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class NAdam:
    # adam + nesterov momentum (ref: arxiv 1609.04747)
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps  # using 1e-7 here, slightly different from adam
        self.m_w = None; self.v_w = None
        self.m_b = None; self.v_b = None
        self.t = 0
    
    def update(self, weights, biases, grads_w, grads_b):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1
        
        for i in range(len(weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1-self.beta1) * grads_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1-self.beta1) * grads_b[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1-self.beta2) * grads_w[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1-self.beta2) * grads_b[i]**2
            
            mw_hat = self.m_w[i] / (1 - self.beta1**self.t)
            mb_hat = self.m_b[i] / (1 - self.beta1**self.t)
            vw_hat = self.v_w[i] / (1 - self.beta2**self.t)
            vb_hat = self.v_b[i] / (1 - self.beta2**self.t)
            
            # nesterov lookahead on the first moment
            nw = self.beta1 * mw_hat + (1-self.beta1) * grads_w[i] / (1 - self.beta1**self.t)
            nb = self.beta1 * mb_hat + (1-self.beta1) * grads_b[i] / (1 - self.beta1**self.t)
            
            weights[i] -= self.lr * nw / (np.sqrt(vw_hat) + self.eps)
            biases[i] -= self.lr * nb / (np.sqrt(vb_hat) + self.eps)


# helper to get optimizer by name
def get_optimizer(name, lr=0.001, **kw):
    opts = {
        'sgd': SGD, 'momentum': MomentumGD, 'nesterov': NesterovGD,
        'rmsprop': RMSProp, 'adam': Adam, 'nadam': NAdam,
    }
    if name.lower() not in opts:
        raise ValueError(f"unknown optimizer: {name}")
    return opts[name.lower()](lr=lr, **kw)
