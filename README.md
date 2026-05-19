# Atri Assignment — Feedforward Neural Network

Feedforward neural network built from scratch using numpy for Fashion-MNIST classification.  
Implements backpropagation with 6 optimizers and uses wandb for experiment tracking.

## Setup

```bash
pip install -r requirements.txt
wandb login
```

## Project structure

- `model.py` — neural network class (forward pass, backprop, loss functions)
- `optimizers.py` — SGD, Momentum, Nesterov, RMSProp, Adam, NAdam
- `train.py` — training loop with wandb integration
- `utils.py` — data loading, preprocessing, helper functions
- `sweep.yaml` — wandb sweep configuration
- `notebooks/` — jupyter notebooks for each question

## How to train

Single run:
```bash
python train.py --epochs 20 --optimizer adam --hidden_size 128 --activation relu
```

With sweep:
```bash
wandb sweep sweep.yaml
wandb agent <sweep_id> --count 50
```

## Data split

Using the standard Fashion-MNIST train/test split from keras:
- Training: 60,000 images → split into 54,000 train + 6,000 validation (10%)
- Test: 10,000 images (never used during training)

The validation set is created by randomly shuffling and splitting the training data with a fixed seed for reproducibility. See `utils.py/train_val_split()`.

## Results

Best model config: Adam optimizer, 3 hidden layers × 128 neurons, ReLU activation, Xavier init

| Dataset | Val Accuracy | Test Accuracy |
|---------|-------------|---------------|
| Fashion-MNIST | 89.17% | 88.67% |
| MNIST | — | 98.02% |

## wandb Report

https://api.wandb.ai/links/vaishalinir-ymc2022-chennai-institute-of-technology/h6rzeth9
