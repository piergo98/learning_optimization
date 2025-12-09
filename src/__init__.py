"""
Optimizers module for gradient-based optimization.

Includes NumPy-based optimizers (SGD, Adam, RMSProp) and PyTorch GPU-accelerated versions.
"""

__version__ = "0.1.0"

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
    from .optimizers import (
        SGD,
        Adam,
        RMSProp,
        gpu_optimize,
    )
    _TORCH_AVAILABLE = True

    # provide common aliases so the imported symbols are actually referenced
    SGD = SGD
    Adam = Adam
    RMSProp = RMSProp
    gpu_optimize = gpu_optimize

    __all__ = [
        '__version__',
        'StochasticGradientDescent', 'sgd_optimize',
        'RMSPropOptimizer', 'AdamOptimizer', 'rmsprop_optimize', 'adam_optimize',
        'SGD', 'Adam', 'RMSProp', 'gpu_optimize',
        '_TORCH_AVAILABLE',
    ]
except ImportError:
    _TORCH_AVAILABLE = False
    __all__ = [
        '__version__',
        'StochasticGradientDescent', 'sgd_optimize',
        'RMSPropOptimizer', 'AdamOptimizer', 'rmsprop_optimize', 'adam_optimize',
        '_TORCH_AVAILABLE',
    ]