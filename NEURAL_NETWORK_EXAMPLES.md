# Neural Network Training Examples

This guide demonstrates how to use the neural network training utilities with custom optimizers.

## Table of Contents

1. [GPU Training with PyTorch](#gpu-training-pytorch)
2. [Multi-class Classification](#multi-class-classification)
3. [Regression Problems](#regression-problems)
4. [Comparing Optimizers](#comparing-optimizers)

## GPU Training (PyTorch)

Leverage GPU acceleration for faster training:

```python
import torch
from optimizers.neural_network import NeuralNetwork, train_neural_network

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Generate data
torch.manual_seed(42)
n_samples = 10000
X_train = torch.randn(n_samples, 20)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()

X_val = torch.randn(2000, 20)
y_val = (X_val[:, 0] + X_val[:, 1] > 0).long()

# Create network
network = NeuralNetwork(
    layer_sizes=[20, 64, 32, 2],
    activation='relu',
    output_activation='softmax'
)

# Train on GPU
params_opt, history = train_neural_network(
    network=network,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    optimizer='adam',
    learning_rate=0.001,
    n_epochs=100,
    batch_size=64,
    device=device,
    verbose=True
)

# Make predictions
network.eval()
with torch.no_grad():
    predictions = network(X_val.to(device))
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes.cpu() == y_val).float().mean().item()
print(f"Validation Accuracy: {accuracy:.4f}")
```

## Multi-class Classification

Train on MNIST-like dataset:

```python
import torch
from optimizers.neural_network import NeuralNetwork, train_neural_network

# Load your MNIST data (or use sklearn.datasets.load_digits for demo)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load digits dataset (8x8 images, 10 classes)
digits = load_digits()
X = digits.data  # 1797 samples, 64 features
y = digits.target

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create network
network = NeuralNetwork(
    layer_sizes=[64, 128, 64, 10],  # 64 inputs -> 10 classes
    activation='relu',
    output_activation='softmax'
)

# Train
params_opt, history = train_neural_network(
    network=network,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    optimizer='adam',
    learning_rate=0.001,
    n_epochs=100,
    batch_size=32,
    device=device,
    verbose=True
)

# Evaluate
predictions = network.forward(X_val, params_opt)
predicted_classes = torch.argmax(predictions, axis=1)
accuracy = torch.mean(predicted_classes == y_val)
print(f"\nFinal Validation Accuracy: {accuracy:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print("\nClassification Report:")
print(classification_report(y_val, predicted_classes))
```

## Regression Problems

Use neural networks for regression with MSE loss:

```python
import torch
from optimizers.neural_network import NeuralNetwork, train_neural_network

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Generate synthetic regression data
torch.random.seed(42)
n_samples = 1000
X_train = torch.random.randn(n_samples, 5)
# Non-linear target function
y_train = (X_train[:, 0]**2 + torch.sin(X_train[:, 1]) + 
           X_train[:, 2] * X_train[:, 3] + torch.random.randn(n_samples) * 0.1)
y_train = y_train.reshape(-1, 1)  # Shape (n_samples, 1)

X_val = torch.random.randn(200, 5)
y_val = (X_val[:, 0]**2 + torch.sin(X_val[:, 1]) + 
         X_val[:, 2] * X_val[:, 3] + torch.random.randn(200) * 0.1)
y_val = y_val.reshape(-1, 1)

# Create network for regression
network = NeuralNetwork(
    layer_sizes=[5, 32, 16, 1],  # 5 inputs -> 1 output
    activation='relu',
    output_activation='linear'  # Linear output for regression
)

# Train
params_opt, history = train_neural_network(
    network=network,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    optimizer='adam',
    learning_rate=0.001,
    n_epochs=100,
    batch_size=32,
    device=device,
    verbose=True
)

# Evaluate
predictions = network.forward(X_val, params_opt)
mse = torch.mean((predictions - y_val)**2)
rmse = torch.sqrt(mse)
print(f"\nValidation RMSE: {rmse:.4f}")
```

## Comparing Optimizers

Compare different optimizers on the same problem:

```python
import torch
import matplotlib.pyplot as plt
from optimizers.neural_network import NeuralNetwork, train_neural_network

# Generate data
torch.random.seed(42)
X_train = torch.random.randn(1000, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Test different optimizers
optimizers = ['sgd', 'adam', 'rmsprop']
histories = {}

for opt_name in optimizers:
    print(f"\n{'='*60}")
    print(f"Training with {opt_name.upper()}")
    print('='*60)
    
    # Create fresh network
    network = NeuralNetwork(
        layer_sizes=[10, 32, 16, 2],
        activation='relu',
        output_activation='softmax'
    )
    
    # Train
    _, history = train_neural_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        optimizer=opt_name,
        learning_rate=0.01 if opt_name == 'sgd' else 0.001,
        n_epochs=50,
        batch_size=32,
        verbose=False
    )
    
    histories[opt_name] = history

# Plot comparison
plt.figure(figsize=(10, 6))
for opt_name, history in histories.items():
    plt.plot(history['loss'], label=opt_name.upper(), linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Optimizer Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150)
plt.show()

print("\nFinal losses:")
for opt_name, history in histories.items():
    print(f"  {opt_name.upper()}: {history['loss'][-1]:.4f}")
```

## Custom Architecture

Build deeper networks with different activation functions:

```python
import torch
from optimizers.neural_network import NeuralNetwork, train_neural_network

# Deep network with tanh activation
network = NeuralNetwork(
    layer_sizes=[100, 256, 128, 64, 32, 10],  # 5 layers
    activation='tanh',  # Try 'relu', 'tanh', 'sigmoid'
    output_activation='softmax'
)

# For very deep networks, consider:
# - Lower learning rate (e.g., 0.0001)
# - Gradient clipping (can be added to optimizer)
# - Batch normalization (would require custom implementation)
# - Smaller batch sizes for better generalization
```

## Saving and Loading Models

Save trained parameters for later use:

```python
import torch

# After training
params_opt, history = train_neural_network(...)

# Save parameters
torch.save('trained_network_params.npy', params_opt)

# Save network configuration
config = {
    'layer_sizes': network.layer_sizes,
    'activation': network.activation,
    'output_activation': network.output_activation
}
torch.save('network_config.npy', config)

# Later: Load and use
config = torch.load('network_config.npy', allow_pickle=True).item()
params = torch.load('trained_network_params.npy')

network_loaded = NeuralNetwork(**config)
predictions = network_loaded.forward(X_test, params)
```

## Tips for Better Performance

1. **Data Preprocessing**:
   - Normalize/standardize inputs (zero mean, unit variance)
   - For images: scale to [0, 1] or [-1, 1]

2. **Learning Rate**:
   - Start with 0.001 for Adam/RMSProp
   - Start with 0.01 for SGD with momentum
   - Use learning rate schedules for better convergence

3. **Batch Size**:
   - Smaller (16-32): Better generalization, more noise
   - Larger (128-256): Faster training, more stable

4. **Network Architecture**:
   - Start simple, add complexity if needed
   - ReLU activation usually works well for hidden layers
   - Softmax for multi-class, sigmoid for binary, linear for regression

5. **Regularization** (to be added):
   - L2 weight decay (available in optimizer)
   - Dropout (requires custom implementation)
   - Early stopping based on validation loss

## Running the Examples

```bash
# Run the built-in examples
cd /path/to/learning_optimization
python -m optimizers.neural_network
```
