# Installation and Development Guide

## Installation

### Standard Installation

Install the package with basic dependencies (NumPy, Matplotlib):

```bash
pip install .
```

Or in development/editable mode (recommended for development):

```bash
pip install -e .
```

### Installation with GPU Support

To use PyTorch GPU-accelerated optimizers:

```bash
pip install .[gpu]
```

### Installation with CasADi Support

To use CasADi for symbolic differentiation:

```bash
pip install .[casadi]
```

### Installation with All Optional Dependencies

```bash
pip install .[all]
```

### Development Installation

Install with development tools (pytest, black, flake8, mypy):

```bash
pip install -e .[dev,all]
```

## Usage After Installation

Once installed, you can import the package from anywhere:

```python
# Import NumPy-based optimizers
from optimizers import StochasticGradientDescent, sgd_optimize
from optimizers import RMSPropOptimizer, AdamOptimizer
from optimizers import rmsprop_optimize, adam_optimize

# Import PyTorch GPU optimizers (if torch is installed)
from optimizers import TorchSGD, TorchAdam, TorchRMSProp, gpu_optimize

# Example usage
import numpy as np

# Define a simple quadratic objective
def gradient_func(params, indices=None, data=None):
    return 2 * params  # gradient of ||params||^2

# Optimize
params_init = np.random.randn(10)
params_opt, history = sgd_optimize(
    params_init=params_init,
    gradient_func=gradient_func,
    optimizer='adam',
    learning_rate=0.01,
    n_epochs=100,
    verbose=True
)
```

## Development Workflow

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=sgd --cov-report=html
```

### Code Formatting

Format code with Black:

```bash
black sgd/
```

### Linting

```bash
flake8 sgd/
```

### Type Checking

```bash
mypy sgd/
```

## Building Distribution

Build source and wheel distributions:

```bash
pip install build
python -m build
```

This creates `dist/learning_optimization-0.1.0.tar.gz` and `dist/learning_optimization-0.1.0-py3-none-any.whl`.

## Publishing to PyPI (Optional)

```bash
pip install twine
twine upload dist/*
```

## Uninstalling

```bash
pip uninstall learning-optimization
```

## Package Structure

```
learning_optimization/
├── optimizers/                  # Main package
│   ├── __init__.py             # Package exports
│   ├── sgd.py                  # NumPy optimizers
│   ├── optimizers.py           # PyTorch optimizers
│   ├── example.py              # Usage examples
│   └── ...
├── pyproject.toml              # Modern package metadata
├── setup.py                    # Legacy setup script
├── README.md                   # Project documentation
├── INSTALL.md                  # This file
└── LICENSE                     # License file
```

## Troubleshooting

### Import Error After Installation

If you get `ModuleNotFoundError: No module named 'optimizers'`, ensure:

1. The package is installed: `pip list | grep learning-optimization`
2. You're using the correct Python environment
3. You're not in the project root with a local `optimizers/` folder conflicting

### PyTorch Import Issues

If you get errors importing PyTorch optimizers but don't need them:

```python
# The package handles missing torch gracefully
from optimizers import sgd_optimize  # Works without torch
# from optimizers import gpu_optimize  # Only works if torch installed
```

### Development Mode Changes Not Reflected

If using `pip install -e .` and changes aren't reflected:

1. Restart your Python interpreter
2. Check you're importing from the editable installation: `import optimizers; print(optimizers.__file__)`
