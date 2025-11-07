# Quick Reference

## Installation Commands

```bash
# Basic installation
pip install -e .

# With GPU support
pip install -e .[gpu]

# With CasADi
pip install -e .[casadi]

# All optional dependencies
pip install -e .[all]

# Development mode with testing tools
pip install -e .[dev,all]
```

## Import Examples

```python
# NumPy-based optimizers
from optimizers import sgd_optimize, adam_optimize, rmsprop_optimize
from optimizers import StochasticGradientDescent, AdamOptimizer, RMSPropOptimizer

# PyTorch GPU optimizers (requires torch)
from optimizers import gpu_optimize
from optimizers import SGD, Adam, RMSProp

# Check if PyTorch is available
from optimizers import _TORCH_AVAILABLE
```

## Basic Usage

### Using convenience functions

```python
import numpy as np
from optimizers import adam_optimize

def my_gradient(params, indices=None, data=None):
    # Your gradient computation
    return gradient_array

params_init = np.random.randn(100)
params_opt, history = adam_optimize(
    params_init=params_init,
    gradient_func=my_gradient,
    learning_rate=0.001,
    n_epochs=1000,
    verbose=True
)
```

### Using optimizer classes

```python
from optimizers import AdamOptimizer

optimizer = AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999
)

params_opt, history = optimizer.optimize(
    params_init=params_init,
    gradient_func=my_gradient,
    n_epochs=1000
)
```

### GPU Optimization

```python
import torch
from optimizers import gpu_optimize

def my_gradient(params, indices=None, data=None):
    # params is torch.Tensor on device
    # return torch.Tensor gradient
    return gradient_tensor

params_init = torch.randn(1000, device='cuda')
params_opt, history = gpu_optimize(
    params_init=params_init,
    gradient_func=my_gradient,
    optimizer='adam',
    learning_rate=0.001,
    n_epochs=1000,
    device='cuda',
    verbose=True
)
```

## Optimizer Selection

```python
# Using sgd_optimize with optimizer selection
from optimizers import sgd_optimize

params_opt, history = sgd_optimize(
    params_init=params_init,
    gradient_func=my_gradient,
    optimizer='adam',  # 'sgd', 'adam', 'rmsprop'
    learning_rate=0.001,
    n_epochs=1000
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optimizers --cov-report=html

# Run specific test file
pytest tests/test_sgd.py -v
```

## Building Distribution

```bash
# Install build tools
pip install build

# Build package
python -m build

# Output in dist/:
# - learning_optimization-0.1.0.tar.gz
# - learning_optimization-0.1.0-py3-none-any.whl
```

## Package Structure

```
learning_optimization/
├── optimizers/                   # Main package
│   ├── __init__.py              # Package exports & version
│   ├── sgd.py                   # NumPy optimizers (SGD, Adam, RMSProp)
│   ├── optimizers.py            # PyTorch GPU optimizers
│   ├── example.py               # NumPy examples
│   ├── torch_example.py         # PyTorch examples
│   └── requirements.txt         # Dependencies
├── tests/                        # Test suite
│   ├── test_import.py
│   └── test_sgd.py
├── pyproject.toml               # Package metadata (modern)
├── setup.py                     # Setup script (legacy compat)
├── MANIFEST.in                  # Distribution files
├── README.md                    # Documentation
├── INSTALL.md                   # Installation guide
└── LICENSE                      # License file
```
