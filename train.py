"""
training script for the feedforward neural network
uses wandb for experiment tracking

usage:
    python train.py --epochs 10 --optimizer adam --hidden_size 128
"""

import numpy as np
import argparse
import wandb
import pickle
import os

from model import NeuralNetwork
from optimizers import get_optimizer
from utils import load_fashion_mnist, one_hot, train_val_split, compute_accuracy


def train(config=None):
    """called by wandb agent during sweep"""
    
    with wandb.init(config=config):
        cfg = wandb.config
        
        epochs = cfg.get('epochs', 10)
        num_layers = cfg.get('num_layers', 3)
        hidden_size = cfg.get('hidden_size', 128)
        weight_decay = cfg.get('weight_decay', 0)
        lr = cfg.get('learning_rate', 0.001)
        opt_name = cfg.get('optimizer', 'adam')
        batch_size = cfg.get('batch_size', 32)
        weight_init = cfg.get('weight_init', 'xavier')
        activation = cfg.get('activation', 'relu')
        loss_fn = cfg.get('loss_type', 'cross_entropy')
        
        run_name = f"hl_{num_layers}_hs_{hidden_size}_opt_{opt_name}_act_{activation}_bs_{batch_size}"
        wandb.run.name = run_name
        
        X_train, y_train, X_test, y_test = load_fashion_mnist()
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.1)
        y_train_oh = one_hot(y_train)
        y_val_oh = one_hot(y_val)
        
        sizes = [784] + [hidden_size] * num_layers + [10]
        model = NeuralNetwork(sizes, activation=activation, weight_init=weight_init)
        optimizer = get_optimizer(opt_name, lr=lr)
        
        n = X_train.shape[0]
        best_val = 0
        
        for ep in range(epochs):
            perm = np.random.permutation(n)
            X_shuf = X_train[perm]
            y_shuf = y_train_oh[perm]
            
            total_loss = 0
            nbatch = 0
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                
                yp, cache = model.forward(xb)
                loss = model.compute_loss(yp, yb, loss_type=loss_fn, weight_decay=weight_decay)
                total_loss += loss
                nbatch += 1
                
                gw, gb = model.backward(yp, yb, cache, loss_type=loss_fn, weight_decay=weight_decay)
                optimizer.update(model.weights, model.biases, gw, gb)
            
            avg_loss = total_loss / nbatch
            
            tr_acc = compute_accuracy(y_train, model.predict(X_train))
            vl_acc = compute_accuracy(y_val, model.predict(X_val))
            vl_probs, _ = model.forward(X_val)
            vl_loss = model.compute_loss(vl_probs, y_val_oh, loss_type=loss_fn, weight_decay=weight_decay)
            
            wandb.log({
                'epoch': ep + 1,
                'train_loss': avg_loss, 'val_loss': vl_loss,
                'train_acc': tr_acc, 'val_acc': vl_acc,
            })
            
            print(f"ep {ep+1}/{epochs} — loss: {avg_loss:.4f}, val_acc: {vl_acc:.4f}")
            
            if vl_acc > best_val:
                best_val = vl_acc
                os.makedirs('models', exist_ok=True)
                with open('models/best_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        test_acc = compute_accuracy(y_test, model.predict(X_test))
        wandb.log({'test_acc': test_acc, 'best_val_acc': best_val})
        print(f"\ntest accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    wandb.init(project='atri-assignment')
    train()
    wandb.finish()
