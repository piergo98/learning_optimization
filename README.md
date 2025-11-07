# learning_optimization

This repository contains algorithms and optimizer implementations for learning-based optimization of control inputs and mode durations in switched optimal control problems (switched OCP).

## Installation

### Quick Install

```bash
pip install -e .
```

### With GPU Support (PyTorch)

```bash
pip install -e .[gpu]
```

### With All Dependencies

```bash
pip install -e .[all]
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Quick Start

```python
from optimizers import sgd_optimize, adam_optimize
import numpy as np

# Define your gradient function
def gradient_func(params, indices=None, data=None):
    # Your gradient computation here
    return 2 * params  # Example: gradient of ||params||^2

# Initialize parameters
params_init = np.random.randn(10)

# Optimize using Adam
params_opt, history = adam_optimize(
    params_init=params_init,
    gradient_func=gradient_func,
    learning_rate=0.01,
    n_epochs=100,
    verbose=True
)
```

## Contents

- `optimizers/` — A flexible optimization library with NumPy-based and PyTorch GPU-accelerated optimizers (SGD, Adam, RMSProp).

Below are the bundled README contents from the `optimizers` subfolder: the plain SGD documentation followed by the PyTorch GPU-accelerated README. This wraps the inner docs so you can read usage and examples from the project root.

---

## Stochastic Gradient Descent (SGD) Implementation

This folder contains a flexible implementation of the Stochastic Gradient Descent algorithm designed to work with pre-computed gradient expressions.

### Features

- Full and Mini-batch SGD: Support for both full-batch and mini-batch gradient descent
- Momentum: Classical momentum and Nesterov accelerated gradient
- Learning Rate Scheduling: Multiple built-in schedules (constant, step, exponential, inverse) and support for custom schedules
- Regularization: L2 weight decay (regularization)
- Early Stopping: Based on gradient norm or loss improvement
- History Tracking: Records loss, learning rate, and gradient norm during optimization

### Files

- `sgd.py`: Main implementation of the SGD algorithm
- `example.py`: Comprehensive examples demonstrating various use cases
- `__init__.py`: Package initialization
- `README.md`: This file (the `sgd` subfolder README)

### Installation

No additional dependencies beyond NumPy and Matplotlib (for visualization in examples).

```bash
pip install numpy matplotlib
```

### Usage

Basic usage and examples are provided in the `sgd` subfolder. The optimizer accepts a gradient function with the signature:

```python
def my_gradient_func(params, indices=None, data=None):
	# params: current parameter vector (numpy.ndarray)
	# indices: optional mini-batch indices
	# data: optional additional data (e.g., X, y)
	gradient = ...  # compute gradient as numpy array
	return gradient
```

Key entry points:

- `StochasticGradientDescent` class: fine-grained control over SGD behavior (momentum, schedules, weight decay)
- `sgd_optimize(...)` convenience function: quick entry point to run SGD

See `sgd/example.py` for complete runnable examples (quadratic, linear regression, schedules, templates).

### Integration with Your Gradient Expressions

If you already have gradient expressions (e.g., from CasADi or symbolic derivations), wrap them in a function matching the `my_gradient_func` signature and pass them to `sgd_optimize` or the optimizer class. The `sgd` implementation expects NumPy arrays in the optimization loop for best performance.

---

## PyTorch GPU-Accelerated Optimizers

This project also includes PyTorch-based implementations of SGD, Adam, and RMSProp that can run on GPU for large-scale optimization problems.

### Why use the PyTorch optimizers?

- GPU acceleration: Leverage CUDA for faster optimization on large problems
- Automatic device management: Seamlessly move tensors between CPU and GPU
- Compatible interface: Same API as NumPy optimizers, just pass `device='cuda'`
- Easy integration: Works with CasADi gradients via simple wrappers

### Installation

Install PyTorch (with CUDA support if you have an NVIDIA GPU):

```bash
# CPU only
pip install torch

# With CUDA (choose the correct wheel for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start (PyTorch)

```python
from sgd import torch_optimize
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def my_gradient(params_torch, indices=None, data=None):
	# compute gradient and return a torch.Tensor on the same device as params_torch
	return grad_tensor

params_init = torch.randn(1000)
params_opt, history = torch_optimize(
	params_init,
	my_gradient,
	optimizer='adam',
	learning_rate=0.01,
	n_epochs=200,
	device=device
)
```

### CasADi integration

CasADi is CPU-only for symbolic evaluation. To use CasADi gradients with the PyTorch optimizers, wrap CasADi calls to convert between NumPy and torch:

```python
def gradient_wrapper(params_torch, indices=None, data=None):
	params_np = params_torch.detach().cpu().numpy()
	grad_casadi = casadi_grad_func(params_np)
	grad_np = np.array(grad_casadi.full()).ravel()
	grad_torch = torch.tensor(grad_np, dtype=torch.float32, device=params_torch.device)
	return grad_torch
```

### Examples

See `sgd/gpu_example.py` for runnable demos (CPU vs GPU comparison, CasADi wrapper example, optimizer comparison, large-scale GPU example).

---

## Final notes

- The `sgd` directory contains both NumPy (CPU) and PyTorch (GPU) optimizers. Use the appropriate implementation depending on problem size and environment.
- For most switched OCP problems with modest parameter counts, the NumPy/CPU implementation is sufficient. Use the PyTorch/GPU path when parameters are large or you run many parallel scenarios.

If you'd like, I can also add a short section here with commands to run the examples or a small benchmark comparing CPU vs GPU on your machine.
