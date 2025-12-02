"""
Neural Network Training with Custom Optimizers

This module provides utilities for training neural networks using the custom
optimizers (SGD, Adam, RMSProp) with PyTorch implementations.

Features:
- Simple feedforward neural network implementation
- Support for custom gradient-based optimizers
- Training utilities and helper functions
- Integration with PyTorch for GPU acceleration
"""
import numpy as np
import os, subprocess, sys
from typing import Optional, Callable, Tuple, Dict, List
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU training will not be available.")
    
from switched_linear_torch import SwiLin
from optimizers import gpu_optimize


# ============================================================================
# PyTorch Implementation
# ============================================================================

if TORCH_AVAILABLE:
    
    class SwiLinNN(nn.Module):
        """
        Simple feedforward neural network using PyTorch to solve a switched linear optimal control problem.
        
        This implementation is compatible with PyTorch GPU optimizers.
        
        Parameters
        ----------
        layer_sizes : list of int
            Number of units in each layer.
        activation : str
            Activation function: 'relu', 'tanh', 'sigmoid'.
        output_activation : str
            Output layer activation: 'softmax', 'sigmoid', 'linear'.
        """
        
        def __init__(
            self,
            layer_sizes: List[int],
            n_phases: int,
            activation: str = 'relu',
            output_activation: str = 'linear'
        ):
            super().__init__()
            
            # Build switched linear problem
            self.n_phases = n_phases
            self.sys, _ = self.switched_problem(self.n_phases)
            
            self.layer_sizes = layer_sizes
            self.activation = activation
            self.output_activation = output_activation

            # Build layers
            self.layers = nn.ModuleList()
            for i in range(len(layer_sizes) - 1):
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                
        def switched_problem(self, n_phases: int):
            """Define switched linear problem."""
            model = {
                'A': [
                    np.array([[-2.5, 0.5, 0.3], [0.4, -2.0, 0.6], [0.2, 0.3, -1.8]]),
                    np.array([[-1.9, 3.2, 0.4], [0.3, -2.1, 0.5], [0, 0.6, -2.3]]),
                    np.array([[-2.2, 0, 0.5],   [0.2, -1.7, 0.4], [0.3, 0.2, -2.0]]),
                    np.array([[-1.8, 0.3, 0.2], [0.5, -2.4, 0],   [0.4, 0, -2.2]]),
                    np.array([[-2.0, 0.4, 0],   [0.3, -2.2, 0.2], [0.5, 0.3, -1.9]]),
                    np.array([[-2.3, 0.2, 0.3], [0, -2.0, 0.4],   [0.2, 0.5, -2.1]]),
                    np.array([[-1.7, 0.5, 0.4], [0.2, -2.5, 0.3], [1.1, 0.2, -2.4]]),
                    np.array([[-2.1, 0.3, 0.2], [0.4, -1.9, 0.5], [0.3, 0.1, -2.0]]),
                    np.array([[-2.4, 0, 0.5],   [0.2, -2.3, 0.3], [0.4, 0.2, -1.8]]),
                    np.array([[-1.8, 0.4, 0.3], [0.5, -2.1, 0.2], [0.2, 3.1, -2.2]]),
                ],
                'B': [
                    np.array([[1.5, 0.3], [0.4, 1.2], [0.2, 0.8]]),
                    np.array([[1.2, 0.5], [0.3, 0.9], [0.4, 1.1]]),
                    np.array([[1.0, 0.4], [0.5, 1.3], [0.3, 0.7]]),
                    np.array([[1.4, 0.2], [0.6, 1.0], [0.1, 0.9]]),
                    np.array([[1.3, 0.1], [0.2, 1.4], [0.5, 0.6]]),
                    np.array([[1.1, 0.3], [0.4, 1.5], [0.2, 0.8]]),
                    np.array([[1.6, 0.2], [0.3, 1.1], [0.4, 0.7]]),
                    np.array([[1.0, 0.4], [0.5, 1.2], [0.3, 0.9]]),
                    np.array([[1.2, 0.5], [0.1, 1.3], [0.6, 0.8]]),
                    np.array([[1.4, 0.3], [0.2, 1.0], [0.5, 0.7]]),
                ],
            }

            n_states = model['A'][0].shape[0]
            n_inputs = model['B'][0].shape[1]

            self.time_horizon = 2

            x0 = np.array([2, -1, 5])
            
            xr = np.array([1, -3])
            
            swi_lin = SwiLin(
                n_phases, 
                n_states,
                n_inputs,
                self.time_horizon, 
                auto=False, 
            )
            
            # Load model
            swi_lin.load_model(model)

            Q = 10. * np.eye(n_states)
            R = 10. * np.eye(n_inputs)
            E = 1. * np.eye(n_states)

            swi_lin.precompute_matrices(x0, Q, R, E)
            x0_aug = np.append(x0, 1)  # augment with 1 for affine term
            
            # Store the cost function (now returns a callable)
            self.cost_func = swi_lin.cost_function(R, sym_x0=True)
            self.R = R
            
            return swi_lin, x0_aug
        
        def forward(self, x):
            """Forward pass."""
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                if i < len(self.layers) - 1:
                    # Hidden layers
                    if self.activation == 'relu':
                        x = F.relu(x)
                    elif self.activation == 'tanh':
                        x = torch.tanh(x)
                    elif self.activation == 'sigmoid':
                        x = torch.sigmoid(x)
                else:
                    # Output layer
                    if self.output_activation == 'softmax':
                        x = F.softmax(x, dim=-1)
                    elif self.output_activation == 'sigmoid':
                        x = torch.sigmoid(x)
                    elif self.output_activation == 'linear':
                        pass  # No activation
            
            return x
        
        def get_flat_params(self):
            """Get flattened network parameters."""
            params = []
            for param in self.parameters():
                params.append(param.data.view(-1))
            return torch.cat(params)
        
        def set_flat_params(self, flat_params):
            """Set network parameters from flattened vector."""
            idx = 0
            for param in self.parameters():
                numel = param.numel()
                param.data = flat_params[idx:idx + numel].view(param.shape)
                idx += numel
    
    
    def train_neural_network(
        network: SwiLinNN,
        X_train: torch.Tensor,
        y_train: Optional[torch.Tensor] = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
        device: str = 'cpu',
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Train neural network using PyTorch GPU optimizers.
        
        Parameters
        ----------
        network : NeuralNetworkPyTorch
            Neural network instance.
        X_train : torch.Tensor, shape (n_samples, n_features)
            Training data.
        y_train : torch.Tensor, shape (n_samples,)
            Training labels.
        X_val : torch.Tensor, optional
            Validation data.
        y_val : torch.Tensor, optional
            Validation labels.
        optimizer : str
            Optimizer name: 'sgd', 'adam', 'rmsprop'.
        learning_rate : float
            Learning rate.
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        device : str
            Device: 'cpu' or 'cuda'.
        verbose : bool
            Print training progress.
            
        Returns
        -------
        params_optimized : torch.Tensor
            Optimized network parameters.
        history : dict
            Training history.
        """
        
        network = network.to(device)
        X_train = X_train.to(device)
        
        if X_val is not None:
            X_val = X_val.to(device)
        
        n_samples = X_train.shape[0]
        n_inputs = network.sys.n_inputs
        
        # Gradient function
        def gradient_func(params, indices=None, data=None):
            """Compute gradient of loss w.r.t. network parameters."""
            network.set_flat_params(params)
            network.zero_grad()
            
            if indices is not None:
                X_batch = X_train[indices]
            else:
                X_batch = X_train
            
            batch_size = X_batch.shape[0]
            output = network(X_batch)
            
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(network.sys.time_horizon, device=output.device, dtype=output.dtype)

            # Handle batch dimension properly
            if output.dim() == 1:
                # Single sample case
                # Output format: [u_1_1, u_1_2, ..., u_n_1, u_n_2, ..., delta_1, ..., delta_n]
                # Controls: first n_phases * n_inputs elements
                # Deltas: last n_phases elements
                n_control_outputs = network.n_phases * n_inputs
                controls = output[:n_control_outputs]
                delta_raw = output[n_control_outputs:]
                
                # Apply softmax and scale deltas
                delta_normalized = F.softmax(delta_raw, dim=-1)
                deltas = delta_normalized * T_tensor
                
                transformed_output = torch.cat([controls, deltas], dim=0)
            else:
                # Batch case
                n_control_outputs = network.n_phases * n_inputs
                controls = output[:, :n_control_outputs]
                delta_raw = output[:, n_control_outputs:]
                
                # Apply softmax and scale deltas
                delta_normalized = F.softmax(delta_raw, dim=-1)
                deltas = delta_normalized * T_tensor
                
                transformed_output = torch.cat([controls, deltas], dim=-1)
            
            # Compute loss for each sample in batch
            total_loss = 0
            for i in range(batch_size):
                # Extract u and delta for this sample
                sample_output = transformed_output[i] if batch_size > 1 else transformed_output
                x0_sample = X_batch[i] if batch_size > 1 else X_batch
                
                # Reshape controls: (n_phases, n_inputs)
                u_flat = sample_output[:network.n_phases * n_inputs]
                u_list = [u_flat[j*n_inputs:(j+1)*n_inputs] for j in range(network.n_phases)]
                
                # Get deltas
                delta_list = [sample_output[network.n_phases * n_inputs + j] for j in range(network.n_phases)]
                
                # Compute cost using the PyTorch cost function
                # cost_func expects: (*u, *delta, x0)
                args = u_list + delta_list + [x0_sample]
                cost = network.cost_func(*args)
                total_loss = total_loss + cost
            
            # Average loss over batch
            loss = total_loss / batch_size
            
            # Backward pass - computes gradients w.r.t. all network parameters
            network.zero_grad()
            loss.backward()
            
            # Collect gradients from all parameters
            grads = []
            for param in network.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
                else:
                    # Handle parameters without gradients (shouldn't happen normally)
                    grads.append(torch.zeros_like(param.view(-1)))
    
            gradient = torch.cat(grads)
            return gradient
        
        # Loss function
        def loss_func(params, data=None):
            """Compute loss for given parameters."""
            network.set_flat_params(params)
            output = network(X_train)
            
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(network.sys.time_horizon, device=output.device, dtype=output.dtype)
            
            batch_size = X_train.shape[0]
            n_control_outputs = network.n_phases * n_inputs
            controls = output[:, :n_control_outputs]
            delta_raw = output[:, n_control_outputs:]
            
            # Apply softmax and scale deltas
            delta_normalized = F.softmax(delta_raw, dim=-1)
            deltas = delta_normalized * T_tensor
            
            transformed_output = torch.cat([controls, deltas], dim=-1)
            
            # Compute loss for each sample in batch
            total_loss = 0
            for i in range(batch_size):
                sample_output = transformed_output[i]
                x0_sample = X_train[i]
                
                # Reshape controls
                u_flat = sample_output[:network.n_phases * n_inputs]
                u_list = [u_flat[j*n_inputs:(j+1)*n_inputs] for j in range(network.n_phases)]
                
                # Get deltas
                delta_list = [sample_output[network.n_phases * n_inputs + j] for j in range(network.n_phases)]
                
                # Compute cost
                args = u_list + delta_list + [x0_sample]
                cost = network.cost_func(*args)
                total_loss = total_loss + cost
            
            return (total_loss / batch_size).item()
        
        # Initial parameters
        params_init = network.get_flat_params()
        
        # Train
        params_optimized, history = gpu_optimize(
            params_init=params_init,
            gradient_func=gradient_func,
            loss_func=loss_func,
            optimizer=optimizer,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_samples=n_samples,
            device=device,
            verbose=verbose
        )
        
        # Set final parameters
        network.set_flat_params(params_optimized)
        
        # Compute final loss
        if verbose:
            with torch.no_grad():
                final_loss = loss_func(params_optimized)
                print(f"\nFinal Training Loss: {final_loss:.6f}")
                
                if X_val is not None:
                    # Compute validation loss
                    X_val_old = X_train
                    X_train = X_val
                    val_loss = loss_func(params_optimized)
                    X_train = X_val_old
                    print(f"Final Validation Loss: {val_loss:.6f}")
        
        return params_optimized, history
    
    
    def compute_accuracy_torch(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy for PyTorch tensors."""
        if predictions.dim() == 2 and predictions.shape[1] > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        if targets.dim() == 2:
            target_classes = torch.argmax(targets, dim=1)
        else:
            target_classes = targets
        
        return (pred_classes == target_classes).float().mean().item()


# ============================================================================
# Example Usage
# ============================================================================


def example_nahs_torch():
    """
    Example: Train neural network on NAHS example using PyTorch optimizer.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping GPU example.")
        return
    
    print("=" * 70)
    print("Example: Neural Network Training")
    print("=" * 70)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    n_phases = 50
    n_control_inputs = 2
    n_NN_inputs = 3
    n_NN_outputs = n_phases * (n_control_inputs + 1)
    
    X_train = torch.empty(n_samples, n_NN_inputs).uniform_(-5.0, 5.0)
    
    X_val = torch.empty(200, n_NN_inputs).uniform_(-5.0, 5.0)
    
    # Create network
    network = SwiLinNN(
        layer_sizes=[n_NN_inputs, 128, 256, n_NN_outputs],
        n_phases=n_phases,
        activation='relu',
        output_activation='linear'
    )
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    params_opt, history = train_neural_network(
        network=network,
        X_train=X_train,
        # y_train=None,
        X_val=X_val,
        # y_val=None,
        optimizer='sgd',
        learning_rate=0.001,
        n_epochs=100,
        batch_size=32,
        device=device,
        verbose=True
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    import time
    
    # print("\n" + "=" * 70 + "\n")
    
    # Run PyTorch example
    start = time.time()
    example_nahs_torch()
    end = time.time()
    print(f"PyTorch example took {end - start:.2f} seconds")
