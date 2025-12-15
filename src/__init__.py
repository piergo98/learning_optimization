"""learning_optimization.src

Top-level package exports for the ``src`` package.

This file exposes the main classes and utilities implemented under ``src/`` so
they can be imported as ``from learning_optimization.src import SwiLin`` etc.

Only existing modules are imported here — previous versions referenced
optimizer modules that are not part of this repository layout and caused
import-time errors. Keep this file minimal and resilient to optional
dependencies (e.g. PyTorch).
"""

__version__ = "0.1.0"

from .switched_linear_torch import SwiLin
from .training import SwiLinNN, train_neural_network
from .validation import ModelValidator
from .switched_system_simulator import SwitchedSystemSimulator

__all__ = [
    '__version__',
    'SwiLin',
    'SwiLinNN', 'train_neural_network',
    'ModelValidator',
    'SwitchedSystemSimulator',
]