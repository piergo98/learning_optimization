"""
Example: Using PyTorch optimizers for GPU-accelerated optimization.

This example shows how to:
1. Use PyTorch optimizers on CPU
2. Use PyTorch optimizers on GPU (if available)
3. Wrap CasADi gradients for use with PyTorch optimizers
4. Compare performance between CPU and GPU
"""

import numpy as np
import torch
import time
from sgd import torch_optimize, TorchAdam, TorchSGD

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def example_1_simple_quadratic():
    """Example 1: Simple quadratic minimization with PyTorch optimizers."""
    print("\n" + "="*60)
    print("Example 1: Quadratic Minimization (CPU vs GPU)")
    print("="*60)
    
    # Problem: minimize f(x) = (x - x_opt)^T @ A @ (x - x_opt)
    n_dim = 100
    torch.manual_seed(42)
    
    A = torch.randn(n_dim, n_dim)
    A = A.T @ A + torch.eye(n_dim)  # Make positive definite
    x_opt = torch.ones(n_dim) * 3.0
    
    def gradient_func_torch(x, indices=None, data=None):
        """Gradient function that works with torch tensors."""
        A_local, x_opt_local = data
        # Ensure tensors are on same device as x
        A_local = A_local.to(x.device)
        x_opt_local = x_opt_local.to(x.device)
        diff = x - x_opt_local
        return 2 * A_local @ diff
    
    def loss_func_torch(x, data=None):
        """Loss function for monitoring."""
        A_local, x_opt_local = data
        A_local = A_local.to(x.device)
        x_opt_local = x_opt_local.to(x.device)
        diff = x - x_opt_local
        return (diff.T @ A_local @ diff).item()
    
    # Initial parameters
    x_init = torch.randn(n_dim)
    
    # CPU optimization
    print("\n--- Running on CPU ---")
    start_time = time.time()
    x_cpu, hist_cpu = torch_optimize(
        x_init,
        gradient_func_torch,
        loss_func=loss_func_torch,
        optimizer='adam',
        learning_rate=0.1,
        n_epochs=100,
        data=(A, x_opt),
        device='cpu',
        verbose=False
    )
    cpu_time = time.time() - start_time
    print(f"Time: {cpu_time:.4f}s")
    print(f"Final loss: {hist_cpu['loss'][-1]:.6e}")
    print(f"Distance from optimum: {torch.norm(x_cpu - x_opt).item():.6e}")
    
    # GPU optimization (if available)
    if device == 'cuda':
        print("\n--- Running on GPU ---")
        start_time = time.time()
        x_gpu, hist_gpu = torch_optimize(
            x_init,
            gradient_func_torch,
            loss_func=loss_func_torch,
            optimizer='adam',
            learning_rate=0.1,
            n_epochs=100,
            data=(A, x_opt),
            device='cuda',
            verbose=False
        )
        gpu_time = time.time() - start_time
        print(f"Time: {gpu_time:.4f}s")
        print(f"Final loss: {hist_gpu['loss'][-1]:.6e}")
        print(f"Distance from optimum: {torch.norm(x_gpu.cpu() - x_opt).item():.6e}")
        print(f"\nSpeedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("\n--- GPU not available ---")


def example_2_casadi_wrapper():
    """Example 2: Wrapping CasADi gradients for PyTorch."""
    print("\n" + "="*60)
    print("Example 2: CasADi Gradient Wrapper")
    print("="*60)
    
    try:
        import casadi as ca
    except ImportError:
        print("CasADi not installed. Skipping this example.")
        return
    
    # Create a CasADi symbolic expression
    n_dim = 10
    x_sym = ca.SX.sym('x', n_dim)
    
    # Simple quadratic: f(x) = x^T @ x
    f = ca.dot(x_sym, x_sym)
    
    # Compute gradient symbolically
    grad_f = ca.jacobian(f, x_sym).T
    
    # Create CasADi function
    casadi_grad_func = ca.Function('grad', [x_sym], [grad_f])
    casadi_loss_func = ca.Function('loss', [x_sym], [f])
    
    def torch_gradient_wrapper(params_torch, indices=None, data=None):
        """Wrapper that converts torch -> numpy -> CasADi -> numpy -> torch."""
        # Convert torch tensor to numpy (move to CPU if on GPU)
        params_np = params_torch.detach().cpu().numpy()
        
        # Evaluate CasADi function
        grad_casadi = casadi_grad_func(params_np)
        
        # Convert CasADi result to numpy
        grad_np = np.array(grad_casadi.full()).ravel()
        
        # Convert back to torch tensor on same device as input
        grad_torch = torch.tensor(grad_np, dtype=torch.float32, device=params_torch.device)
        
        return grad_torch
    
    def torch_loss_wrapper(params_torch, data=None):
        """Loss wrapper for monitoring."""
        params_np = params_torch.detach().cpu().numpy()
        loss_casadi = casadi_loss_func(params_np)
        return float(loss_casadi.full().squeeze())
    
    # Run optimization
    x_init = torch.randn(n_dim) * 5.0  # Start far from optimum
    
    print("\nOptimizing with Adam (using CasADi gradients)...")
    x_opt, history = torch_optimize(
        x_init,
        torch_gradient_wrapper,
        loss_func=torch_loss_wrapper,
        optimizer='adam',
        learning_rate=0.1,
        n_epochs=50,
        device=device,
        verbose=True
    )
    
    print(f"\nFinal parameters norm: {torch.norm(x_opt).item():.6e}")
    print("(Should be close to 0)")


def example_3_optimizer_comparison():
    """Example 3: Compare different PyTorch optimizers."""
    print("\n" + "="*60)
    print("Example 3: Optimizer Comparison on GPU" if device == 'cuda' else "Example 3: Optimizer Comparison on CPU")
    print("="*60)
    
    # Rosenbrock-like function (more challenging)
    n_dim = 50
    
    def gradient_func(x, indices=None, data=None):
        """Gradient of sum of squared differences."""
        grad = torch.zeros_like(x)
        grad[:-1] = -2 * (1 - x[:-1]) + 400 * x[:-1] * (x[:-1]**2 - x[1:])
        grad[1:] += 200 * (x[1:] - x[:-1]**2)
        return grad
    
    def loss_func(x, data=None):
        """Rosenbrock-like function."""
        return (torch.sum((1 - x[:-1])**2 + 100 * (x[1:] - x[:-1]**2)**2)).item()
    
    x_init = torch.zeros(n_dim)
    
    optimizers_to_test = [
        ('SGD', {'momentum': 0.9}),
        ('Adam', {'beta1': 0.9, 'beta2': 0.999}),
        ('RMSProp', {'rho': 0.9}),
    ]
    
    results = {}
    
    for opt_name, opt_params in optimizers_to_test:
        print(f"\n--- {opt_name} ---")
        start_time = time.time()
        
        x_final, hist = torch_optimize(
            x_init.clone(),
            gradient_func,
            loss_func=loss_func,
            optimizer=opt_name.lower(),
            learning_rate=0.001 if opt_name == 'SGD' else 0.01,
            n_epochs=200,
            device=device,
            verbose=False,
            **opt_params
        )
        
        elapsed = time.time() - start_time
        results[opt_name] = {
            'time': elapsed,
            'final_loss': hist['loss'][-1],
            'history': hist
        }
        
        print(f"Time: {elapsed:.4f}s")
        print(f"Final loss: {hist['loss'][-1]:.6e}")
        print(f"Final gradient norm: {hist['gradient_norm'][-1]:.6e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for opt_name, res in results.items():
        print(f"{opt_name:10s}: Loss={res['final_loss']:12.6e}  Time={res['time']:.4f}s")


def example_4_large_scale():
    """Example 4: Large-scale problem to showcase GPU benefits."""
    print("\n" + "="*60)
    print("Example 4: Large-Scale Problem (GPU Advantage)")
    print("="*60)
    
    if device != 'cuda':
        print("GPU not available. Skipping large-scale example.")
        return
    
    # Very large problem
    n_dim = 10000
    print(f"Problem size: {n_dim} parameters")
    
    torch.manual_seed(42)
    A = torch.randn(n_dim, n_dim, device='cuda')
    A = A.T @ A / n_dim + torch.eye(n_dim, device='cuda')
    x_opt = torch.ones(n_dim, device='cuda')
    
    def gradient_func(x, indices=None, data=None):
        A_local, x_opt_local = data
        diff = x - x_opt_local
        return 2 * A_local @ diff
    
    def loss_func(x, data=None):
        A_local, x_opt_local = data
        diff = x - x_opt_local
        return (diff.T @ A_local @ diff).item()
    
    x_init = torch.randn(n_dim, device='cuda')
    
    print("\nRunning Adam on GPU...")
    start_time = time.time()
    x_final, history = torch_optimize(
        x_init,
        gradient_func,
        loss_func=loss_func,
        optimizer='adam',
        learning_rate=0.01,
        n_epochs=100,
        data=(A, x_opt),
        device='cuda',
        verbose=False
    )
    gpu_time = time.time() - start_time
    
    print(f"Time: {gpu_time:.4f}s")
    print(f"Final loss: {history['loss'][-1]:.6e}")
    print(f"Throughput: {n_dim * 100 / gpu_time:.0f} param-updates/sec")


if __name__ == "__main__":
    # Run all examples
    example_1_simple_quadratic()
    example_2_casadi_wrapper()
    example_3_optimizer_comparison()
    example_4_large_scale()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
