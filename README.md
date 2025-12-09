# learning_optimization

This repository contains algorithms and optimizer implementations for learning-based optimization of control inputs and mode durations in switched optimal control problems (switched OCP).

## Installation

```bash
pip install -e .
```

For optional GPU support (PyTorch), install with the `gpu` extras:

```bash
pip install -e .[gpu]
```

With All Dependencies

```bash
pip install -e .[all]
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Quick Start

### Neural Network Training (PyTorch)

```python
from src.training import SwiLinNN, train_neural_network
import torch

network = SwiLinNN(layer_sizes=[3, 128, 256, 100], n_phases=50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
params_opt, history = train_neural_network(
    network=network,
    X_train=torch.randn(100, 3),
    optimizer='adam',
    learning_rate=1e-3,
    n_epochs=50,
    device=device
)
```

## Package Layout

- `src/` — core Python package containing the optimization code and examples.
  - `src/training.py` — PyTorch neural network utilities and training helpers (`SwiLinNN`, `train_neural_network`).
  - `src/switched_linear_torch.py` — PyTorch-native switched linear utilities (`SwiLin`) that compute cost and support autograd.
- `examples/` — runnable examples demonstrating usage.
- [NEURAL_NETWORK_EXAMPLES.md](NEURAL_NETWORK_EXAMPLES.md) — detailed neural-network-specific examples and instructions.

## Switched Linear Module

The switched-linear MPC utilities are implemented natively in PyTorch in `src/switched_linear_torch.py`.

- `SwiLin.precompute_matrices(x0, Q, R, E)`: precompute numeric matrices required by the cost.
- `SwiLin.cost_function(R, sym_x0=True)`: returns a callable cost function accepting `(u_1, ..., u_N, delta_1, ..., delta_N, x0)` and returns a PyTorch scalar (compatible with autograd).

This removes the previous dependency on CasADi code-generation wrappers — the code is now PyTorch-native and supports automatic differentiation directly.

