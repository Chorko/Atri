import numpy as np
import matplotlib.pyplot as plt

# fashion mnist class names
FASHION_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_fashion_mnist():
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # flatten and normalize
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    
    return X_train, y_train, X_test, y_test


def one_hot(y, num_classes=10):
    n = y.shape[0]
    out = np.zeros((n, num_classes))
    out[np.arange(n), y] = 1
    return out

# TODO: add train/val split and accuracy metric
