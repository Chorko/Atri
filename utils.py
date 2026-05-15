import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# from sklearn.metrics import confusion_matrix  # might use this later

# fashion mnist class names
FASHION_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

MNIST_LABELS = [str(i) for i in range(10)]


def load_fashion_mnist():
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # flatten and normalize
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    
    print(f"loaded fashion mnist — train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_mnist():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    return X_train, y_train, X_test, y_test


def one_hot(y, num_classes=10):
    # TODO: maybe use keras.utils.to_categorical instead?
    n = y.shape[0]
    out = np.zeros((n, num_classes))
    out[np.arange(n), y] = 1
    return out


def train_val_split(X, y, val_ratio=0.1, seed=42):
    """keep 10% of training data for validation (assignment requirement)"""
    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.permutation(n)
    val_size = int(n * val_ratio)
    
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    # print so we can verify the split is done properly
    print(f"train/val split — train: {len(train_idx)}, val: {len(val_idx)} (ratio={val_ratio})")
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def plot_samples(X, y, labels=FASHION_LABELS):
    """for Q1 — plot one image from each class"""
    # reshape if flattened
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        idx = np.where(y == i)[0][0]
        ax.imshow(X[idx], cmap='gray')
        ax.set_title(labels[i])
        ax.axis('off')
    
    plt.suptitle('Fashion MNIST - One Sample Per Class', fontsize=14)
    plt.tight_layout()
    plt.show()


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# run this file directly to check data loading works
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train)
    print(f"shapes — X_tr: {X_tr.shape}, X_val: {X_val.shape}")
    print(f"labels distribution — train: {np.bincount(y_tr)}")
    print(f"labels distribution — val: {np.bincount(y_val)}")
