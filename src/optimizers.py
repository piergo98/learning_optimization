"""
PyTorch-based optimizers for GPU-accelerated optimization.

These optimizers work with torch.Tensor objects and can run on GPU for large-scale problems.
All optimizers follow the same interface as the NumPy versions but operate on PyTorch tensors.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple, Dict
import warnings


def _check_finite_tensor(x: torch.Tensor, context: str = '') -> None:
    """Raise if tensor contains NaN or Inf; include detailed diagnostics.

    The message includes: shape, dtype, device, number of finite/non-finite elements,
    min/max/norm over finite elements, indices of the first non-finite elements
    (up to 10), and a small value sample for quick inspection.
    """
    if not torch.all(torch.isfinite(x)):
        # attempt to compute diagnostics on CPU
        x_cpu = x.detach().cpu()
        finite_mask = torch.isfinite(x_cpu)
        num_total = x_cpu.numel()
        num_finite = int(finite_mask.sum().item())
        num_nonfinite = num_total - num_finite

        any_finite = num_finite > 0
        try:
            x_min = float(torch.min(x_cpu[finite_mask]).item()) if any_finite else float('nan')
            x_max = float(torch.max(x_cpu[finite_mask]).item()) if any_finite else float('nan')
            x_norm = float(torch.norm(x_cpu[finite_mask]).item()) if any_finite else float('nan')
        except Exception:
            x_min = x_max = x_norm = float('nan')

        # indices of first few non-finite elements
        try:
            nonfinite_idx = (~finite_mask).nonzero(as_tuple=False).flatten()[:10].cpu().numpy().tolist()
        except Exception:
            nonfinite_idx = []

        # sample some elements (first 10)
        sample = x_cpu.flatten()[:10].cpu().numpy().tolist()

        raise ValueError(
            f"Non-finite values detected in {context}. "
            f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}. "
            f"Finite/Total={num_finite}/{num_total} (non-finite={num_nonfinite}). "
            f"Stats over finite elements: min={x_min}, max={x_max}, norm={x_norm}. "
            f"First non-finite indices (up to 10): {nonfinite_idx}. "
            f"Sample( first 10 values ): {sample}"
        )


class SGD:
    """
    PyTorch-based SGD optimizer with GPU support.
    
    Parameters
    ----------
    learning_rate : float
        Initial learning rate.
    momentum : float
        Momentum factor (0 <= momentum < 1).
    nesterov : bool
        Whether to use Nesterov momentum.
    weight_decay : float
        L2 regularization coefficient.
    device : str or torch.device
        Device to run optimization on ('cpu', 'cuda', 'cuda:0', etc.).
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        
        self.velocity = None
        self.history = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }
    
    def _initialize_velocity(self, params: torch.Tensor) -> None:
        """Initialize velocity for momentum."""
        if self.velocity is None:
            self.velocity = torch.zeros_like(params, device=self.device)
    
    def step(self, params: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """
        Perform a single SGD update step.
        
        Parameters
        ----------
        params : torch.Tensor
            Current parameters.
        gradient : torch.Tensor
            Gradient of the loss with respect to parameters.
            
        Returns
        -------
        torch.Tensor
            Updated parameters.
        """
        # Ensure tensors are on the correct device
        params = params.to(self.device)
        gradient = gradient.to(self.device)
        
        self._initialize_velocity(params)
        
        # Add weight decay to gradient
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * params
        
        # Update velocity and parameters
        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity - self.lr * gradient
            
            if self.nesterov:
                params = params + self.momentum * self.velocity - self.lr * gradient
            else:
                params = params + self.velocity
        else:
            params = params - self.lr * gradient
        
        return params
    
    def optimize(
        self,
        params_init: Union[torch.Tensor, np.ndarray],
        gradient_func: Callable,
        loss_func: Optional[Callable] = None,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        data: Optional[Tuple] = None,
        n_samples: Optional[int] = None,
        verbose: bool = True,
        tol: float = 1e-6,
        patience: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run the SGD optimization.
        
        Parameters
        ----------
        params_init : torch.Tensor or np.ndarray
            Initial parameters (will be converted to torch.Tensor on device).
        gradient_func : callable
            Function to compute gradients. Should accept torch.Tensor and return torch.Tensor.
            Signature: gradient_func(params, indices=None, data=None) -> gradient
        loss_func : callable, optional
            Function to compute loss. Should accept torch.Tensor and return scalar.
        n_epochs : int
            Number of epochs.
        batch_size : int, optional
            Mini-batch size.
        data : tuple, optional
            Data to pass to gradient_func and loss_func.
        n_samples : int, optional
            Total number of samples.
        verbose : bool
            Whether to print progress.
        tol : float
            Tolerance for early stopping.
        patience : int, optional
            Patience for early stopping.
            
        Returns
        -------
        params : torch.Tensor
            Optimized parameters.
        history : dict
            Optimization history.
        """
        # Convert to torch tensor if needed
        if not isinstance(params_init, torch.Tensor):
            params = torch.tensor(params_init, dtype=torch.float32, device=self.device)
        else:
            params = params_init.clone().to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.history = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }
        
        for epoch in range(n_epochs):
            # Generate mini-batches
            if batch_size is not None and n_samples is not None:
                indices = torch.randperm(n_samples, device=self.device)
                n_batches = int(torch.ceil(torch.tensor(n_samples / batch_size)))
            else:
                indices = None
                n_batches = 1
            
            epoch_gradients = []
            
            for batch_idx in range(n_batches):
                # Get batch indices
                if indices is not None:
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                else:
                    batch_indices = None
                
                # Compute gradient
                gradient = gradient_func(params, indices=batch_indices, data=data)
                
                # Ensure gradient is a tensor on correct device
                if not isinstance(gradient, torch.Tensor):
                    gradient = torch.tensor(gradient, dtype=torch.float32, device=self.device)
                else:
                    gradient = gradient.to(self.device)

                # Sanity check: gradient must be finite. If not, include parameter stats to help debug.
                try:
                    _check_finite_tensor(gradient, context=f'gradient (epoch={epoch}, batch={batch_idx})')
                except ValueError as e:
                    # capture a few useful parameter statistics for debugging
                    try:
                        params_cpu = params.detach().cpu().numpy()
                        p_min = float(np.min(params_cpu))
                        p_max = float(np.max(params_cpu))
                        p_mean = float(np.mean(params_cpu))
                        p_norm = float(np.linalg.norm(params_cpu))
                        params_sample = params_cpu.flatten()[:10].tolist()
                        raise ValueError(
                            str(e) +
                            f"\nParameter stats at failure: shape={params_cpu.shape}, "
                            f"min={p_min}, max={p_max}, mean={p_mean}, norm={p_norm}. "
                            f"Param sample (first 10): {params_sample}"
                        )
                    except Exception:
                        # If params can't be inspected, re-raise original
                        raise
                
                epoch_gradients.append(gradient)
                
                # Update parameters
                params = self.step(params, gradient)
            
            # Compute average gradient norm
            avg_gradient = torch.stack(epoch_gradients).mean(dim=0)
            gradient_norm = torch.norm(avg_gradient).item()
            
            # Compute loss
            if loss_func is not None:
                loss = loss_func(params, data=data)
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                # Sanity check: loss should be finite
                if not np.isfinite(loss):
                    raise ValueError(f"Non-finite loss detected at epoch {epoch}: {loss}")
                self.history['loss'].append(loss)
                
                # Early stopping based on loss
                if patience is not None:
                    if loss < best_loss - tol:
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                loss = None
            
            self.history['learning_rate'].append(self.lr)
            self.history['gradient_norm'].append(gradient_norm)
            
            # Verbose output
            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                msg = f"Epoch {epoch+1}/{n_epochs} - LR: {self.lr:.6f} - Grad norm: {gradient_norm:.6e}"
                if loss is not None:
                    msg += f" - Loss: {loss:.6e}"
                print(msg)
            
            # Early stopping based on gradient norm
            if gradient_norm < tol:
                if verbose:
                    print(f"Converged at epoch {epoch+1} (gradient norm < {tol})")
                break
        
        return params, self.history


class Adam:
    """
    PyTorch-based Adam optimizer with GPU support.
    
    Parameters
    ----------
    learning_rate : float
        Learning rate.
    beta1 : float
        Exponential decay rate for first moment.
    beta2 : float
        Exponential decay rate for second moment.
    eps : float
        Small constant for numerical stability.
    weight_decay : float
        L2 regularization coefficient.
    device : str or torch.device
        Device to run on.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        
        self.m = None
        self.v = None
        self.t = 0
        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}
    
    def _init_state(self, params: torch.Tensor):
        if self.m is None:
            self.m = torch.zeros_like(params, device=self.device)
        if self.v is None:
            self.v = torch.zeros_like(params, device=self.device)
    
    def step(self, params: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """Perform a single Adam update step."""
        params = params.to(self.device)
        gradient = gradient.to(self.device)
        
        self._init_state(params)
        self.t += 1
        
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * params
        
        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params = params - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        return params
    
    def optimize(self, *args, **kwargs):
        """Run Adam optimization (same interface as TorchSGD)."""
        params_init = args[0]
        gradient_func = args[1]
        
        # Extract kwargs
        loss_func = kwargs.get('loss_func', None)
        n_epochs = kwargs.get('n_epochs', 100)
        batch_size = kwargs.get('batch_size', None)
        data = kwargs.get('data', None)
        n_samples = kwargs.get('n_samples', None)
        verbose = kwargs.get('verbose', True)
        tol = kwargs.get('tol', 1e-6)
        patience = kwargs.get('patience', None)
        
        # Convert to torch tensor
        if not isinstance(params_init, torch.Tensor):
            params = torch.tensor(params_init, dtype=torch.float32, device=self.device)
        else:
            params = params_init.clone().to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}
        
        for epoch in range(n_epochs):
            if batch_size is not None and n_samples is not None:
                indices = torch.randperm(n_samples, device=self.device)
                n_batches = int(torch.ceil(torch.tensor(n_samples / batch_size)))
            else:
                indices = None
                n_batches = 1
            
            epoch_gradients = []
            
            for batch_idx in range(n_batches):
                if indices is not None:
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                else:
                    batch_indices = None
                
                gradient = gradient_func(params, indices=batch_indices, data=data)
                
                if not isinstance(gradient, torch.Tensor):
                    gradient = torch.tensor(gradient, dtype=torch.float32, device=self.device)
                else:
                    gradient = gradient.to(self.device)

                # Sanity check: gradient must be finite. If not, include parameter stats to help debug.
                try:
                    _check_finite_tensor(gradient, context=f'gradient (epoch={epoch}, batch={batch_idx})')
                except ValueError as e:
                    try:
                        params_cpu = params.detach().cpu().numpy()
                        p_min = float(np.min(params_cpu))
                        p_max = float(np.max(params_cpu))
                        p_mean = float(np.mean(params_cpu))
                        p_norm = float(np.linalg.norm(params_cpu))
                        params_sample = params_cpu.flatten()[:10].tolist()
                        raise ValueError(
                            str(e) +
                            f"\nParameter stats at failure: shape={params_cpu.shape}, "
                            f"min={p_min}, max={p_max}, mean={p_mean}, norm={p_norm}. "
                            f"Param sample (first 10): {params_sample}"
                        )
                    except Exception:
                        raise

                epoch_gradients.append(gradient)
                params = self.step(params, gradient)
            
            avg_gradient = torch.stack(epoch_gradients).mean(dim=0)
            gradient_norm = torch.norm(avg_gradient).item()
            
            if loss_func is not None:
                loss = loss_func(params, data=data)
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                # Sanity check: loss should be finite
                if not np.isfinite(loss):
                    raise ValueError(f"Non-finite loss detected at epoch {epoch}: {loss}")
                self.history['loss'].append(loss)
                
                if patience is not None:
                    if loss < best_loss - tol:
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                loss = None
            
            self.history['learning_rate'].append(self.lr)
            self.history['gradient_norm'].append(gradient_norm)
            
            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                msg = f"Epoch {epoch+1}/{n_epochs} - LR: {self.lr:.6f} - Grad norm: {gradient_norm:.6e}"
                if loss is not None:
                    msg += f" - Loss: {loss:.6e}"
                print(msg)
            
            if gradient_norm < tol:
                if verbose:
                    print(f"Converged at epoch {epoch+1} (gradient norm < {tol})")
                break
        
        return params, self.history


class RMSProp:
    """
    PyTorch-based RMSProp optimizer with GPU support.
    
    Parameters
    ----------
    learning_rate : float
        Learning rate.
    rho : float
        Decay rate for squared gradient moving average.
    eps : float
        Small constant for numerical stability.
    momentum : float
        Optional momentum factor.
    weight_decay : float
        L2 regularization coefficient.
    device : str or torch.device
        Device to run on.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        
        self.squared_avg = None
        self.velocity = None
        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}
    
    def _init_state(self, params: torch.Tensor):
        if self.squared_avg is None:
            self.squared_avg = torch.zeros_like(params, device=self.device)
        if self.velocity is None:
            self.velocity = torch.zeros_like(params, device=self.device)
    
    def step(self, params: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """Perform a single RMSProp update step."""
        params = params.to(self.device)
        gradient = gradient.to(self.device)
        
        self._init_state(params)
        
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * params
        
        # Update squared gradient average
        self.squared_avg = self.rho * self.squared_avg + (1 - self.rho) * (gradient ** 2)
        
        # Compute adjusted gradient
        adjusted_grad = gradient / (torch.sqrt(self.squared_avg) + self.eps)
        
        # Apply momentum if specified
        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity + self.lr * adjusted_grad
            params = params - self.velocity
        else:
            params = params - self.lr * adjusted_grad
        
        return params
    
    def optimize(self, *args, **kwargs):
        """Run RMSProp optimization (same interface as TorchSGD)."""
        params_init = args[0]
        gradient_func = args[1]
        
        loss_func = kwargs.get('loss_func', None)
        n_epochs = kwargs.get('n_epochs', 100)
        batch_size = kwargs.get('batch_size', None)
        data = kwargs.get('data', None)
        n_samples = kwargs.get('n_samples', None)
        verbose = kwargs.get('verbose', True)
        tol = kwargs.get('tol', 1e-6)
        patience = kwargs.get('patience', None)
        
        if not isinstance(params_init, torch.Tensor):
            params = torch.tensor(params_init, dtype=torch.float32, device=self.device)
        else:
            params = params_init.clone().to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}
        
        for epoch in range(n_epochs):
            if batch_size is not None and n_samples is not None:
                indices = torch.randperm(n_samples, device=self.device)
                n_batches = int(torch.ceil(torch.tensor(n_samples / batch_size)))
            else:
                indices = None
                n_batches = 1
            
            epoch_gradients = []
            
            for batch_idx in range(n_batches):
                if indices is not None:
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                else:
                    batch_indices = None
                
                gradient = gradient_func(params, indices=batch_indices, data=data)
                
                if not isinstance(gradient, torch.Tensor):
                    gradient = torch.tensor(gradient, dtype=torch.float32, device=self.device)
                else:
                    gradient = gradient.to(self.device)
                
                epoch_gradients.append(gradient)
                params = self.step(params, gradient)
            
            avg_gradient = torch.stack(epoch_gradients).mean(dim=0)
            gradient_norm = torch.norm(avg_gradient).item()
            
            if loss_func is not None:
                loss = loss_func(params, data=data)
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                self.history['loss'].append(loss)
                
                if patience is not None:
                    if loss < best_loss - tol:
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                loss = None
            
            self.history['learning_rate'].append(self.lr)
            self.history['gradient_norm'].append(gradient_norm)
            
            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                msg = f"Epoch {epoch+1}/{n_epochs} - LR: {self.lr:.6f} - Grad norm: {gradient_norm:.6e}"
                if loss is not None:
                    msg += f" - Loss: {loss:.6e}"
                print(msg)
            
            if gradient_norm < tol:
                if verbose:
                    print(f"Converged at epoch {epoch+1} (gradient norm < {tol})")
                break
        
        return params, self.history


def gpu_optimize(
    params_init,
    gradient_func: Callable,
    loss_func: Optional[Callable] = None,
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    data: Optional[Tuple] = None,
    n_samples: Optional[int] = None,
    device: Union[str, torch.device] = 'cpu',
    verbose: bool = True,
    tol: float = 1e-6,
    patience: Optional[int] = None,
    **optimizer_kwargs
) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function for PyTorch-based optimization.
    
    Parameters
    ----------
    params_init : array-like or torch.Tensor
        Initial parameters.
    gradient_func : callable
        Gradient function (should accept and return torch.Tensor).
    loss_func : callable, optional
        Loss function.
    optimizer : str
        Optimizer name: 'sgd', 'adam', or 'rmsprop'.
    learning_rate : float
        Learning rate.
    n_epochs : int
        Number of epochs.
    batch_size : int, optional
        Mini-batch size.
    data : tuple, optional
        Data to pass to functions.
    n_samples : int, optional
        Total number of samples.
    device : str or torch.device
        Device ('cpu', 'cuda', 'cuda:0', etc.).
    verbose : bool
        Print progress.
    tol : float
        Convergence tolerance.
    patience : int, optional
        Early stopping patience.
    **optimizer_kwargs
        Additional optimizer-specific parameters.
        
    Returns
    -------
    params : torch.Tensor
        Optimized parameters.
    history : dict
        Optimization history.
    """
    opt_name = optimizer.lower()
    
    if opt_name == 'sgd':
        opt = SGD(
            learning_rate=learning_rate,
            device=device,
            **optimizer_kwargs
        )
    elif opt_name == 'adam':
        opt = Adam(
            learning_rate=learning_rate,
            device=device,
            **optimizer_kwargs
        )
    elif opt_name == 'rmsprop':
        opt = RMSProp(
            learning_rate=learning_rate,
            device=device,
            **optimizer_kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Supported: 'sgd', 'adam', 'rmsprop'")
    
    return opt.optimize(
        params_init,
        gradient_func,
        loss_func=loss_func,
        n_epochs=n_epochs,
        batch_size=batch_size,
        data=data,
        n_samples=n_samples,
        verbose=verbose,
        tol=tol,
        patience=patience
    )
