"""
Stochastic Gradient Descent (SGD) Implementation

This module provides a flexible implementation of the Stochastic Gradient Descent
algorithm that can work with pre-computed gradient expressions.
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple, Dict, List
import warnings


def _check_finite_array(x: np.ndarray, context: str = '') -> None:
    """Raise a clear error if array contains NaN or Inf.

    Includes a small diagnostics message (min/max/norm) to help debugging.
    """
    if not np.all(np.isfinite(x)):
        # compute some diagnostics where possible
        try:
            finite_mask = np.isfinite(x)
            any_finite = np.any(finite_mask)
            x_min = np.min(x[finite_mask]) if any_finite else float('nan')
            x_max = np.max(x[finite_mask]) if any_finite else float('nan')
            x_norm = np.linalg.norm(x[finite_mask]) if any_finite else float('nan')
        except Exception:
            x_min = x_max = x_norm = float('nan')
        raise ValueError(
            f"Non-finite values detected in {context}. "
            f"Stats over finite elements: min={x_min}, max={x_max}, norm={x_norm}. "
            f"Full array sample: {x.flatten()[:10]}"
        )


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent optimizer.
    
    This class implements SGD with support for:
    - Mini-batch training
    - Learning rate scheduling
    - Momentum
    - Nesterov accelerated gradient
    - Weight decay (L2 regularization)
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Initial learning rate for parameter updates.
    momentum : float, default=0.0
        Momentum factor (0 <= momentum < 1).
    nesterov : bool, default=False
        Whether to use Nesterov momentum.
    weight_decay : float, default=0.0
        L2 regularization coefficient.
    learning_rate_schedule : str or callable, optional
        Learning rate schedule. Options: 'constant', 'step', 'exponential', 'inverse'
        or a custom callable with signature schedule(epoch, initial_lr) -> lr.
    schedule_params : dict, optional
        Parameters for the learning rate schedule.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        learning_rate_schedule: Optional[Union[str, Callable]] = 'constant',
        schedule_params: Optional[Dict] = None
    ):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.learning_rate_schedule = learning_rate_schedule
        self.schedule_params = schedule_params or {}
        
        # Velocity for momentum
        self.velocity = None
        
        # History tracking
        self.history = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }
        
    def _initialize_velocity(self, params: np.ndarray) -> None:
        """Initialize velocity for momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
    
    def _update_learning_rate(self, epoch: int) -> None:
        """Update learning rate based on schedule."""
        if callable(self.learning_rate_schedule):
            self.lr = self.learning_rate_schedule(epoch, self.initial_lr)
        elif self.learning_rate_schedule == 'constant':
            self.lr = self.initial_lr
        elif self.learning_rate_schedule == 'step':
            # Reduce LR by gamma every step_size epochs
            step_size = self.schedule_params.get('step_size', 10)
            gamma = self.schedule_params.get('gamma', 0.1)
            self.lr = self.initial_lr * (gamma ** (epoch // step_size))
        elif self.learning_rate_schedule == 'exponential':
            # Exponential decay: lr = initial_lr * exp(-decay_rate * epoch)
            decay_rate = self.schedule_params.get('decay_rate', 0.01)
            self.lr = self.initial_lr * np.exp(-decay_rate * epoch)
        elif self.learning_rate_schedule == 'inverse':
            # Inverse decay: lr = initial_lr / (1 + decay_rate * epoch)
            decay_rate = self.schedule_params.get('decay_rate', 0.01)
            self.lr = self.initial_lr / (1 + decay_rate * epoch)
        else:
            warnings.warn(f"Unknown schedule '{self.learning_rate_schedule}', using constant.")
            self.lr = self.initial_lr
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform a single SGD update step.
        
        Parameters
        ----------
        params : np.ndarray
            Current parameters.
        gradient : np.ndarray
            Gradient of the loss with respect to parameters.
            
        Returns
        -------
        np.ndarray
            Updated parameters.
        """
        # Initialize velocity if needed
        self._initialize_velocity(params)
        
        # Add weight decay to gradient (L2 regularization)
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * params
        
        # Update velocity and parameters
        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity - self.lr * gradient
            
            if self.nesterov:
                # Nesterov accelerated gradient
                params = params + self.momentum * self.velocity - self.lr * gradient
            else:
                # Standard momentum
                params = params + self.velocity
        else:
            # Standard SGD without momentum
            params = params - self.lr * gradient
        
        return params
    
    def optimize(
        self,
        params_init: np.ndarray,
        gradient_func: Callable,
        loss_func: Optional[Callable] = None,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        data: Optional[Tuple] = None,
        n_samples: Optional[int] = None,
        verbose: bool = True,
        tol: float = 1e-6,
        patience: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run the SGD optimization.
        
        Parameters
        ----------
        params_init : np.ndarray
            Initial parameters.
        gradient_func : callable
            Function to compute gradients. Should have signature:
            gradient_func(params, indices=None, **kwargs) -> gradient
            where indices specify the mini-batch samples.
        loss_func : callable, optional
            Function to compute loss for monitoring. Should have signature:
            loss_func(params, **kwargs) -> loss
        n_epochs : int, default=100
            Number of epochs to train.
        batch_size : int, optional
            Size of mini-batches. If None, uses full batch.
        data : tuple, optional
            Data to pass to gradient_func and loss_func.
        n_samples : int, optional
            Total number of samples in dataset. Required if batch_size is provided.
        verbose : bool, default=True
            Whether to print progress.
        tol : float, default=1e-6
            Tolerance for early stopping based on gradient norm.
        patience : int, optional
            Number of epochs with no improvement to wait before early stopping.
            
        Returns
        -------
        params : np.ndarray
            Optimized parameters.
        history : dict
            Dictionary containing optimization history.
        """
        params = params_init.copy()
        best_loss = float('inf')
        patience_counter = 0
        
        # Reset history
        self.history = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }
        
        for epoch in range(n_epochs):
            # Update learning rate
            self._update_learning_rate(epoch)
            
            # Generate mini-batches
            if batch_size is not None and n_samples is not None:
                # Shuffle indices for this epoch
                indices = np.random.permutation(n_samples)
                n_batches = int(np.ceil(n_samples / batch_size))
            else:
                # Full batch
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
                
                # Compute gradient for this batch
                gradient = gradient_func(params, indices=batch_indices, data=data)
                # Sanity check: gradient must be finite numeric array
                try:
                    _check_finite_array(np.asarray(gradient), context='gradient (batch)')
                except ValueError:
                    # Re-raise with context about epoch/batch
                    raise
                epoch_gradients.append(gradient)
                
                # Update parameters
                params = self.step(params, gradient)
            
            # Compute average gradient norm for this epoch
            avg_gradient = np.mean(epoch_gradients, axis=0)
            gradient_norm = np.linalg.norm(avg_gradient)
            
            # Compute loss if loss function provided
            if loss_func is not None:
                loss = loss_func(params, data=data)
                # Sanity check: loss should be finite scalar
                try:
                    loss_val = float(loss)
                except Exception:
                    raise ValueError(f"Loss function returned non-scalar or non-convertible value: {loss}")
                if not np.isfinite(loss_val):
                    raise ValueError(f"Non-finite loss detected at epoch {epoch}: {loss_val}")
                # append loss to history
                self.history['loss'].append(loss_val)
                
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
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self.velocity = None
        self.lr = self.initial_lr
        self.history = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }


class RMSPropOptimizer:
    """RMSProp optimizer implementation (numpy-based).

    Parameters
    ----------
    learning_rate : float
        Base learning rate.
    rho : float
        Decay rate for squared gradient moving average (default 0.9).
    eps : float
        Small epsilon to avoid division by zero.
    momentum : float
        Optional momentum term (classical), default 0.0.
    weight_decay : float
        L2 regularization coefficient.
    learning_rate_schedule : str or callable, optional
        Learning rate schedule (same options as SGD class).
    """

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, eps: float = 1e-8,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 learning_rate_schedule: Optional[Union[str, Callable]] = 'constant',
                 schedule_params: Optional[Dict] = None):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate_schedule = learning_rate_schedule
        self.schedule_params = schedule_params or {}

        self.squared_avg = None
        self.velocity = None
        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}

    def _update_learning_rate(self, epoch: int) -> None:
        if callable(self.learning_rate_schedule):
            self.lr = self.learning_rate_schedule(epoch, self.initial_lr)
        elif self.learning_rate_schedule == 'constant':
            self.lr = self.initial_lr
        elif self.learning_rate_schedule == 'step':
            step_size = self.schedule_params.get('step_size', 10)
            gamma = self.schedule_params.get('gamma', 0.1)
            self.lr = self.initial_lr * (gamma ** (epoch // step_size))
        elif self.learning_rate_schedule == 'exponential':
            decay_rate = self.schedule_params.get('decay_rate', 0.01)
            self.lr = self.initial_lr * np.exp(-decay_rate * epoch)
        elif self.learning_rate_schedule == 'inverse':
            decay_rate = self.schedule_params.get('decay_rate', 0.01)
            self.lr = self.initial_lr / (1 + decay_rate * epoch)
        else:
            warnings.warn(f"Unknown schedule '{self.learning_rate_schedule}', using constant.")
            self.lr = self.initial_lr

    def _init_state(self, params: np.ndarray):
        if self.squared_avg is None:
            self.squared_avg = np.zeros_like(params)
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self._init_state(params)

        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * params

        # update squared average
        self.squared_avg = self.rho * self.squared_avg + (1 - self.rho) * (gradient ** 2)

        # compute update
        adjusted_grad = gradient / (np.sqrt(self.squared_avg) + self.eps)

        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity + self.lr * adjusted_grad
            params = params - self.velocity
        else:
            params = params - self.lr * adjusted_grad

        return params

    def optimize(self, *args, **kwargs):
        # Reuse existing SGD-like optimize loop for convenience by delegating
        # to a small wrapper that calls step. Implemented here to match API.
        params_init = args[0]
        gradient_func = args[1]
        # Extract common kwargs
        n_epochs = kwargs.get('n_epochs', 100)
        batch_size = kwargs.get('batch_size', None)
        data = kwargs.get('data', None)
        n_samples = kwargs.get('n_samples', None)
        loss_func = kwargs.get('loss_func', None)
        verbose = kwargs.get('verbose', True)
        tol = kwargs.get('tol', 1e-6)
        patience = kwargs.get('patience', None)

        params = params_init.copy()
        best_loss = float('inf')
        patience_counter = 0

        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}

        for epoch in range(n_epochs):
            self._update_learning_rate(epoch)

            if batch_size is not None and n_samples is not None:
                indices = np.random.permutation(n_samples)
                n_batches = int(np.ceil(n_samples / batch_size))
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
                epoch_gradients.append(gradient)
                params = self.step(params, gradient)

            avg_gradient = np.mean(epoch_gradients, axis=0)
            gradient_norm = np.linalg.norm(avg_gradient)

            if loss_func is not None:
                loss = loss_func(params, data=data)
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


class AdamOptimizer:
    """Adam optimizer implementation (numpy-based).

    Parameters
    ----------
    learning_rate : float
        Base step size.
    beta1 : float
        Exponential decay rate for first moment estimates.
    beta2 : float
        Exponential decay rate for second moment estimates.
    eps : float
        Small epsilon to prevent division by zero.
    weight_decay : float
        L2 regularization coefficient.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 learning_rate_schedule: Optional[Union[str, Callable]] = 'constant',
                 schedule_params: Optional[Dict] = None):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.learning_rate_schedule = learning_rate_schedule
        self.schedule_params = schedule_params or {}

        self.m = None
        self.v = None
        self.t = 0
        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}

    def _update_learning_rate(self, epoch: int) -> None:
        if callable(self.learning_rate_schedule):
            self.lr = self.learning_rate_schedule(epoch, self.initial_lr)
        elif self.learning_rate_schedule == 'constant':
            self.lr = self.initial_lr
        elif self.learning_rate_schedule == 'step':
            step_size = self.schedule_params.get('step_size', 10)
            gamma = self.schedule_params.get('gamma', 0.1)
            self.lr = self.initial_lr * (gamma ** (epoch // step_size))
        elif self.learning_rate_schedule == 'exponential':
            decay_rate = self.schedule_params.get('decay_rate', 0.01)
            self.lr = self.initial_lr * np.exp(-decay_rate * epoch)
        elif self.learning_rate_schedule == 'inverse':
            decay_rate = self.schedule_params.get('decay_rate', 0.01)
            self.lr = self.initial_lr / (1 + decay_rate * epoch)
        else:
            warnings.warn(f"Unknown schedule '{self.learning_rate_schedule}', using constant.")
            self.lr = self.initial_lr

    def _init_state(self, params: np.ndarray):
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

    def step(self, params: np.ndarray, gradient: np.ndarray, delta_mask: Optional[np.ndarray] = None, time_horizon: Optional[float] = None) -> np.ndarray:
        """
        This method performs a single Adam update step.
        Parameters
        ----------
        params : np.ndarray
            Current parameters.
        gradient : np.ndarray
            Gradient of the loss with respect to parameters.
        Returns
        -------
        np.ndarray
            Updated parameters.
        """
        self._init_state(params)
        self.t += 1

        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * params

        # Update biased first moment estimate (mean of gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        # Update biased second moment estimate (uncentered variance of gradients)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Compute bias-corrected first and second moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # Update parameters
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # Since params are organized as \theta = [u, delta], we can project delta to the constraint set
        if delta_mask is not None:
            n_phases = int(np.sum(delta_mask))
            delta = delta_mask * params
            # Project to non-negative
            delta = np.maximum(delta, 0)
            # Also ensure that the sum of delta equals total_time_horizon
            sum_delta = np.sum(delta)
            delta = delta * (time_horizon / sum_delta)
        
            # Reconstruct params
            params = params * (1 - delta_mask) + delta
            
        return params

    def optimize(self, *args, **kwargs):
        # Same loop pattern as RMSProp.optimize
        params_init = args[0]
        gradient_func = args[1]
        n_epochs = kwargs.get('n_epochs', 100)
        batch_size = kwargs.get('batch_size', None)
        data = kwargs.get('data', None)
        n_samples = kwargs.get('n_samples', None)
        loss_func = kwargs.get('loss_func', None)
        verbose = kwargs.get('verbose', True)
        tol = kwargs.get('tol', 1e-6)
        patience = kwargs.get('patience', None)
        delta_mask = kwargs.get('delta_mask', None)
        time_horizon = kwargs.get('time_horizon', None)

        params = params_init.copy()
        best_loss = float('inf')
        patience_counter = 0

        self.history = {'loss': [], 'learning_rate': [], 'gradient_norm': []}

        for epoch in range(n_epochs):
            self._update_learning_rate(epoch)

            if batch_size is not None and n_samples is not None:
                indices = np.random.permutation(n_samples)
                n_batches = int(np.ceil(n_samples / batch_size))
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
                
                # Compute gradient for this batch
                gradient = gradient_func(params, indices=batch_indices, data=data)
                epoch_gradients.append(gradient)
                # Update parameters using the computed gradient
                params = self.step(params, gradient, delta_mask=delta_mask, time_horizon=time_horizon)

            avg_gradient = np.mean(epoch_gradients, axis=0)
            gradient_norm = np.linalg.norm(avg_gradient)

            if loss_func is not None:
                loss = loss_func(params, data=data)
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


def sgd_optimize(
    params_init: np.ndarray,
    gradient_func: Callable,
    loss_func: Optional[Callable] = None,
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    data: Optional[Tuple] = None,
    n_samples: Optional[int] = None,
    momentum: float = 0.0,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    learning_rate_schedule: Optional[Union[str, Callable]] = 'constant',
    schedule_params: Optional[Dict] = None,
    verbose: bool = True,
    tol: float = 1e-6,
    patience: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for running SGD optimization.
    
    This function creates an SGD optimizer and runs the optimization.
    See StochasticGradientDescent class for parameter descriptions.
    
    Returns
    -------
    params : np.ndarray
        Optimized parameters.
    history : dict
        Dictionary containing optimization history.
    """
    optimizer = StochasticGradientDescent(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
        learning_rate_schedule=learning_rate_schedule,
        schedule_params=schedule_params
    )
    
    return optimizer.optimize(
        params_init=params_init,
        gradient_func=gradient_func,
        loss_func=loss_func,
        n_epochs=n_epochs,
        batch_size=batch_size,
        data=data,
        n_samples=n_samples,
        verbose=verbose,
        tol=tol,
        patience=patience
    )


def rmsprop_optimize(
    params_init: np.ndarray,
    gradient_func: Callable,
    loss_func: Optional[Callable] = None,
    learning_rate: float = 0.001,
    rho: float = 0.9,
    eps: float = 1e-8,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    learning_rate_schedule: Optional[Union[str, Callable]] = 'constant',
    schedule_params: Optional[Dict] = None,
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    data: Optional[Tuple] = None,
    n_samples: Optional[int] = None,
    verbose: bool = True,
    tol: float = 1e-6,
    patience: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convenience wrapper for RMSProp optimization."""
    optimizer = RMSPropOptimizer(
        learning_rate=learning_rate,
        rho=rho,
        eps=eps,
        momentum=momentum,
        weight_decay=weight_decay,
        learning_rate_schedule=learning_rate_schedule,
        schedule_params=schedule_params,
    )

    return optimizer.optimize(
        params_init,
        gradient_func,
        n_epochs=n_epochs,
        batch_size=batch_size,
        data=data,
        n_samples=n_samples,
        loss_func=loss_func,
        verbose=verbose,
        tol=tol,
        patience=patience,
    )


def adam_optimize(
    params_init: np.ndarray,
    gradient_func: Callable,
    loss_func: Optional[Callable] = None,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    learning_rate_schedule: Optional[Union[str, Callable]] = 'constant',
    schedule_params: Optional[Dict] = None,
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    data: Optional[Tuple] = None,
    n_samples: Optional[int] = None,
    verbose: bool = True,
    tol: float = 1e-6,
    patience: Optional[int] = None,
    delta_mask: Optional[np.ndarray] = None,
    time_horizon: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convenience wrapper for Adam optimization."""
    optimizer = AdamOptimizer(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        learning_rate_schedule=learning_rate_schedule,
        schedule_params=schedule_params,
    )

    return optimizer.optimize(
        params_init,
        gradient_func,
        n_epochs=n_epochs,
        batch_size=batch_size,
        data=data,
        n_samples=n_samples,
        loss_func=loss_func,
        verbose=verbose,
        tol=tol,
        patience=patience,
        delta_mask=delta_mask,
        time_horizon=time_horizon,
    )
