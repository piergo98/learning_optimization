# PyTorch GPU-Accelerated Optimizers

This module provides PyTorch-based implementations of SGD, Adam, and RMSProp that can run on GPU for large-scale optimization problems.

## Why PyTorch Optimizers?

- **GPU acceleration**: Leverage CUDA for faster optimization on large problems
- **Automatic device management**: Seamlessly move tensors between CPU and GPU
- **Compatible interface**: Same API as NumPy optimizers, just pass `device='cuda'`
- **Easy integration**: Works with CasADi gradients via simple wrappers

## Installation

Install PyTorch (with CUDA support if you have an NVIDIA GPU):

```bash
# CPU only
pip install torch

# With CUDA 11.8 (check your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# With CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check installation:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Quick Start

### Basic Usage (CPU)

```python
import torch
from sgd import torch_optimize

# Your gradient function (must accept/return torch.Tensor)
def my_gradient(params, indices=None, data=None):
    # Compute gradient...
    return grad_tensor

# Initial parameters
params_init = torch.randn(100)

# Optimize on CPU
params_opt, history = torch_optimize(
    params_init,
    my_gradient,
    optimizer='adam',
    learning_rate=0.01,
    n_epochs=100,
    device='cpu'
)
```

### GPU Acceleration

```python
# Simply change device to 'cuda'
params_opt, history = torch_optimize(
    params_init,
    my_gradient,
    optimizer='adam',
    learning_rate=0.01,
    n_epochs=100,
    device='cuda'  # <-- Run on GPU!
)

# Result is a torch.Tensor on GPU
print(params_opt.device)  # cuda:0

# Move back to CPU if needed
params_cpu = params_opt.cpu().numpy()
```

### Using with CasADi Gradients

Since CasADi doesn't run on GPU, wrap your CasADi functions to convert between NumPy and PyTorch:

```python
import casadi as ca
import numpy as np
import torch
from sgd import torch_optimize

# Create CasADi gradient function
x = ca.SX.sym('x', n)
f = # ... your cost function
grad_f = ca.jacobian(f, x).T
grad_func_casadi = ca.Function('grad', [x], [grad_f])

# Wrapper that handles conversions
def gradient_wrapper(params_torch, indices=None, data=None):
    # torch -> numpy (CPU)
    params_np = params_torch.detach().cpu().numpy()
    
    # Evaluate CasADi (on CPU)
    grad_casadi = grad_func_casadi(params_np)
    grad_np = np.array(grad_casadi.full()).ravel()
    
    # numpy -> torch (on same device as input)
    grad_torch = torch.tensor(
        grad_np, 
        dtype=torch.float32, 
        device=params_torch.device
    )
    return grad_torch

# Use with GPU optimizer
params_opt, history = torch_optimize(
    params_init,
    gradient_wrapper,
    optimizer='adam',
    device='cuda',
    n_epochs=200
)
```

## Available Optimizers

### TorchSGD
```python
from sgd import TorchSGD

opt = TorchSGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0001,
    device='cuda'
)
params_opt, history = opt.optimize(params_init, gradient_func, n_epochs=100)
```

### TorchAdam
```python
from sgd import TorchAdam

opt = TorchAdam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    device='cuda'
)
params_opt, history = opt.optimize(params_init, gradient_func, n_epochs=100)
```

### TorchRMSProp
```python
from sgd import TorchRMSProp

opt = TorchRMSProp(
    learning_rate=0.001,
    rho=0.9,
    eps=1e-8,
    momentum=0.0,
    weight_decay=0.0,
    device='cuda'
)
params_opt, history = opt.optimize(params_init, gradient_func, n_epochs=100)
```

## Convenience Function

Use `torch_optimize` to select optimizer by name:

```python
from sgd import torch_optimize

# Adam
params_opt, history = torch_optimize(
    params_init, gradient_func,
    optimizer='adam',
    learning_rate=0.001,
    beta1=0.9, beta2=0.999,  # Adam-specific params
    device='cuda',
    n_epochs=200
)

# RMSProp
params_opt, history = torch_optimize(
    params_init, gradient_func,
    optimizer='rmsprop',
    learning_rate=0.001,
    rho=0.9,  # RMSProp-specific
    device='cuda',
    n_epochs=200
)

# SGD with momentum
params_opt, history = torch_optimize(
    params_init, gradient_func,
    optimizer='sgd',
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True,
    device='cuda',
    n_epochs=200
)
```

## When to Use GPU

**GPU is beneficial when:**
- Parameter vectors have >10k-100k elements
- Running many epochs (>1000)
- Large batch sizes
- Multiple optimization runs (ensemble methods)

**CPU is fine when:**
- Small problems (<1000 parameters)
- Few epochs (<100)
- CasADi gradient computation dominates (bottleneck is symbolic evaluation, not optimizer)

**Rule of thumb:** For typical switched OCP problems with ~10-100 parameters per phase, CPU is usually sufficient. GPU helps with:
- Large neural network parameter optimization
- Batch optimization over many scenarios
- Ensemble/population-based methods

## Performance Tips

1. **Keep data on GPU**: Move data tensors to GPU once, not every iteration
   ```python
   A_gpu = A.to('cuda')
   data_gpu = (A_gpu, b_gpu)
   ```

2. **Batch operations**: Process multiple scenarios in parallel on GPU
   ```python
   # Instead of looping over scenarios on CPU
   # Stack them and process on GPU
   params_batch = torch.stack([p1, p2, ..., pN]).to('cuda')
   ```

3. **Profile first**: Use `time.time()` or `torch.cuda.Event()` to measure
   ```python
   import time
   start = time.time()
   params_opt, _ = torch_optimize(..., device='cuda')
   torch.cuda.synchronize()  # Wait for GPU to finish
   print(f"Time: {time.time() - start:.4f}s")
   ```

4. **Mixed precision** (advanced): For very large problems
   ```python
   with torch.cuda.amp.autocast():
       # Optimizer runs in mixed precision (FP16/FP32)
       params_opt = opt.step(params, gradient)
   ```

## Examples

Run the comprehensive examples:

```bash
cd sgd/examples
python torch_example.py
```

This will demonstrate:
1. CPU vs GPU comparison on quadratic problem
2. CasADi gradient wrapper
3. Optimizer comparison (SGD, Adam, RMSProp)
4. Large-scale problem showing GPU advantage

## Troubleshooting

**Import error: "torch" could not be resolved**
- Install PyTorch: `pip install torch`

**CUDA out of memory**
- Reduce problem size or batch size
- Use CPU instead: `device='cpu'`
- Clear cache: `torch.cuda.empty_cache()`

**Slow on GPU (slower than CPU)**
- Problem too small: overhead of GPU transfer dominates
- Use CPU for problems with <10k parameters

**CasADi gradients not working on GPU**
- CasADi runs on CPU only
- Use wrapper pattern shown above (convert torch->numpy->casadi->numpy->torch)

## Integration with Your Switched OCP

For your switched optimal control problem:

```python
# Build CasADi gradient as before
swi_lin_mpc.precompute_matrices(x0, Q, R, E)
du, d_delta = swi_lin_mpc.grad_cost_function(k, R)

# Create CasADi functions
grad_func_casadi = ca.Function('grad', [...], [grad_expr])

# Wrap for PyTorch
def torch_grad_wrapper(params_torch, indices=None, data=None):
    params_np = params_torch.detach().cpu().numpy()
    grad = grad_func_casadi(params_np)
    return torch.tensor(np.array(grad.full()).ravel(), 
                       dtype=torch.float32, 
                       device=params_torch.device)

# Optimize on GPU
params_init = torch.randn(n_params)
params_opt, history = torch_optimize(
    params_init,
    torch_grad_wrapper,
    optimizer='adam',
    learning_rate=0.01,
    n_epochs=500,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Convert back to numpy for use with your system
params_final = params_opt.cpu().numpy()
```

## See Also

- NumPy optimizers: `sgd/sgd.py` (CPU-only, no PyTorch dependency)
- Main README: `sgd/README.md`
- Examples: `sgd/examples/`
