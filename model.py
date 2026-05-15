import numpy as np


class NeuralNetwork:
    
    def __init__(self, layer_sizes, activation='relu', weight_init='xavier', seed=42):
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
                # plain random
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        
        # print(f"initialized {self.num_layers}-layer network: {layer_sizes}")
        # FIXME: should probably not seed in __init__ but whatever
    
    def _activate(self, z):
        if self.activation == 'sigmoid':
            z = np.clip(z, -500, 500)  # avoid overflow
            return 1.0 / (1.0 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"unknown activation: {self.activation}")
    
    def _activate_deriv(self, z):
        # derivatives for each activation
        # had to re-derive these from scratch since we cant use autograd
        if self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'relu':
            return (z > 0).astype(float)
    
    def _softmax(self, z):
        # subtract max for numerical stability
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """returns predictions and cache needed for backprop"""
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
    
    def compute_loss(self, y_pred, y_true, loss_type='cross_entropy', weight_decay=0):
        n = y_true.shape[0]
        
        if loss_type == 'cross_entropy':
            eps = 1e-9
            loss = -np.sum(y_true * np.log(y_pred + eps)) / n
        elif loss_type == 'squared_error':
            loss = np.sum((y_true - y_pred) ** 2) / (2 * n)
        else:
            raise ValueError(f"invalid loss: {loss_type}")
        
        # L2 regularization
        if weight_decay > 0:
            l2 = sum(np.sum(W ** 2) for W in self.weights)
            loss += (weight_decay / 2) * l2
        
        return loss
    
    def backward(self, y_pred, y_true, cache, loss_type='cross_entropy', weight_decay=0):
        """backprop — this was the hardest part to get right"""
        n = y_true.shape[0]
        grads_w = []
        grads_b = []
        
        # output layer gradient
        if loss_type == 'cross_entropy':
            # softmax + CE simplifies to this (see andrew ng notes)
            delta = (y_pred - y_true) / n
        elif loss_type == 'squared_error':
            # this one is trickier with softmax output
            diff = y_pred - y_true
            delta = (diff * y_pred * (1 - y_pred)) / n
            # not the exact jacobian but prof said its ok for our case
        else:
            raise ValueError(f"loss type {loss_type} not supported")
        
        # go backwards through layers
        for i in range(self.num_layers - 1, -1, -1):
            a_prev = cache['a'][i]
            
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            if weight_decay > 0:
                dW += weight_decay * self.weights[i]
            
            grads_w.insert(0, dW)
            grads_b.insert(0, db)
            
            if i > 0:
                delta = delta @ self.weights[i].T
                delta = delta * self._activate_deriv(cache['z'][i-1])
        
        return grads_w, grads_b
    
    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# gradient checking — run this to verify backprop is correct
def gradient_check(model, X, y_onehot, eps=1e-5):
    y_pred, cache = model.forward(X)
    grads_w, _ = model.backward(y_pred, y_onehot, cache)
    
    print("running gradient check...")
    W = model.weights[0]
    dW = grads_w[0]
    
    max_err = 0
    # only check a few random weights
    for _ in range(15):
        i = np.random.randint(W.shape[0])
        j = np.random.randint(W.shape[1])
        
        old = W[i, j]
        
        W[i, j] = old + eps
        y_p, _ = model.forward(X)
        loss_p = model.compute_loss(y_p, y_onehot)
        
        W[i, j] = old - eps
        y_m, _ = model.forward(X)
        loss_m = model.compute_loss(y_m, y_onehot)
        
        W[i, j] = old  # restore
        
        num_grad = (loss_p - loss_m) / (2 * eps)
        ana_grad = dW[i, j]
        
        denom = max(abs(num_grad), abs(ana_grad), 1e-8)
        err = abs(num_grad - ana_grad) / denom
        max_err = max(max_err, err)
    
    print(f"max relative error: {max_err:.2e}")
    if max_err < 1e-5:
        print("PASSED")
    else:
        print("FAILED — check your backprop")
    return max_err
