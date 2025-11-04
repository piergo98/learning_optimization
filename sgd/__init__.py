"""
Stochastic Gradient Descent (SGD) module for optimization.
"""

from .sgd import (
    StochasticGradientDescent,
    sgd_optimize,
    RMSPropOptimizer,
    AdamOptimizer,
    rmsprop_optimize,
    adam_optimize,
)

# PyTorch optimizers (optional - only if torch is installed)
try:
    from .torch_optimizers import (
        TorchSGD,
        TorchAdam,
        TorchRMSProp,
        torch_optimize,
    )
    _TORCH_AVAILABLE = True
    __all__ = [
        'StochasticGradientDescent', 'sgd_optimize',
        'RMSPropOptimizer', 'AdamOptimizer', 'rmsprop_optimize', 'adam_optimize',
        'TorchSGD', 'TorchAdam', 'TorchRMSProp', 'torch_optimize',
    ]
except ImportError:
    _TORCH_AVAILABLE = False
    __all__ = [
        'StochasticGradientDescent', 'sgd_optimize',
        'RMSPropOptimizer', 'AdamOptimizer', 'rmsprop_optimize', 'adam_optimize'
    ]