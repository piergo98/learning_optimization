# Stochastic Gradient Descent (SGD) Implementation

This folder contains a flexible implementation of the Stochastic Gradient Descent algorithm designed to work with pre-computed gradient expressions.

## Features

- **Full and Mini-batch SGD**: Support for both full-batch and mini-batch gradient descent
- **Momentum**: Classical momentum and Nesterov accelerated gradient
- **Learning Rate Scheduling**: Multiple built-in schedules (constant, step, exponential, inverse) and support for custom schedules
- **Regularization**: L2 weight decay (regularization)
- **Early Stopping**: Based on gradient norm or loss improvement
- **History Tracking**: Records loss, learning rate, and gradient norm during optimization

## Files

- `sgd.py`: Main implementation of the SGD algorithm and variants (Adam, RMSProp)
- `optimizers.py`: PyTorch GPU-accelerated optimizers
- `neural_network.py`: Neural network training utilities with custom optimizers
- `example.py`: Comprehensive examples demonstrating various use cases
- `torch_example.py`: PyTorch GPU examples
- `__init__.py`: Package initialization
- `README.md`: This file

## Installation

No additional dependencies beyond NumPy and Matplotlib (for visualization in examples).

```bash
pip install numpy matplotlib
```

## Usage

### Basic Usage

```python
from optimizers import sgd_optimize
import numpy as np

# Define your gradient function
def my_gradient_func(params, indices=None, data=None):
    # Your gradient computation here
    # params: current parameter values
    # indices: mini-batch indices (optional)
    # data: any additional data (optional)
    gradient = # ... your gradient expression
    return gradient

# Define loss function (optional, for monitoring)
def my_loss_func(params, data=None):
    loss = # ... your loss computation
    return loss

# Initial parameters
params_init = np.random.randn(10)

# Run optimization
params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    loss_func=my_loss_func,
    learning_rate=0.01,
    n_epochs=100,
    verbose=True
)
```

### Using the Class Interface

```python
from optimizers import StochasticGradientDescent

# Create optimizer
optimizer = StochasticGradientDescent(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True,
    learning_rate_schedule='exponential',
    schedule_params={'decay_rate': 0.01}
)

# Run optimization
params_optimized, history = optimizer.optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    loss_func=my_loss_func,
    n_epochs=100
)
```

### Mini-batch Training

```python
# For stochastic gradient descent with mini-batches
params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    loss_func=my_loss_func,
    learning_rate=0.01,
    n_epochs=100,
    batch_size=32,
    n_samples=1000,  # total number of samples in dataset
    data=(X, y)  # your data
)
```

### Using Momentum

```python
# Standard momentum
params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    learning_rate=0.01,
    momentum=0.9,
    n_epochs=100
)

# Nesterov accelerated gradient
params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True,
    n_epochs=100
)
```

### Learning Rate Schedules

```python
# Step decay
params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    learning_rate=0.1,
    learning_rate_schedule='step',
    schedule_params={'step_size': 20, 'gamma': 0.5},
    n_epochs=100
)

# Exponential decay
params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    learning_rate=0.1,
    learning_rate_schedule='exponential',
    schedule_params={'decay_rate': 0.05},
    n_epochs=100
)

# Custom schedule
def custom_schedule(epoch, initial_lr):
    return initial_lr / (1 + 0.01 * epoch**2)

params_optimized, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient_func,
    learning_rate=0.1,
    learning_rate_schedule=custom_schedule,
    n_epochs=100
)
```

## Integration with Your Gradient Expressions

Since you mentioned you already have gradient expressions from another code, here's how to integrate them:

1. **Wrap your gradient computation in a function** with the required signature:
   ```python
   def gradient_func(params, indices=None, data=None):
       # Extract parameters if needed
       control_inputs = params[:n_controls]
       mode_durations = params[n_controls:]
       
       # Use your gradient expressions
       grad_controls = your_control_gradient_expression(control_inputs, mode_durations)
       grad_durations = your_duration_gradient_expression(control_inputs, mode_durations)
       
       # Combine into single gradient vector
       gradient = np.concatenate([grad_controls, grad_durations])
       
       # If using mini-batches, select only batch samples
       if indices is not None:
           gradient = compute_batch_gradient(gradient, indices)
       
       return gradient
   ```

2. **Pass this function to SGD** along with your initial parameters.

## Examples

Run the examples to see the optimizer in action:

```bash
cd sgd
python3 example.py
```

This will run four examples:
1. Quadratic function minimization
2. Linear regression with mini-batch SGD
3. Comparison of learning rate schedules
4. Template for custom gradient integration

## Parameters Reference

### `StochasticGradientDescent` / `sgd_optimize`

- **learning_rate** (float): Initial learning rate
- **momentum** (float): Momentum factor (0 ≤ momentum < 1)
- **nesterov** (bool): Use Nesterov momentum
- **weight_decay** (float): L2 regularization coefficient
- **learning_rate_schedule** (str or callable): Learning rate schedule
  - Options: `'constant'`, `'step'`, `'exponential'`, `'inverse'`, or custom function
- **schedule_params** (dict): Parameters for the learning rate schedule
- **n_epochs** (int): Number of training epochs
- **batch_size** (int): Mini-batch size (None for full batch)
- **n_samples** (int): Total number of samples (required with batch_size)
- **verbose** (bool): Print progress during optimization
- **tol** (float): Gradient norm tolerance for early stopping
- **patience** (int): Epochs to wait for improvement before early stopping

## History Output

The `history` dictionary returned by `optimize()` contains:
- **loss**: List of loss values at each epoch (if loss_func provided)
- **learning_rate**: List of learning rates at each epoch
- **gradient_norm**: List of gradient norms at each epoch

## Tips for Optimal Control Problems

For switched OCP with control inputs and mode durations:

1. **Parameter vector structure**: Organize your parameters as a single vector (e.g., `[u_1, ..., u_n, τ_1, ..., τ_m]`)

2. **Gradient computation**: Your gradient function should return gradients in the same order

3. **Constraints**: SGD doesn't handle constraints directly. Consider:
   - Using projected gradient descent (project after each SGD step)
   - Parameterizing to satisfy constraints automatically
   - Adding penalty terms to your loss function

4. **Initialization**: Good initialization is crucial. Use domain knowledge or a simpler method first.

5. **Learning rate**: Start with a small learning rate (0.001-0.01) and use learning rate schedules for better convergence.

## License

This code is part of the learning_optimization repository.
