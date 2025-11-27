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
import casadi as ca
import numpy as np
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
    
from ocslc.switched_linear_mpc import SwiLin
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
            activation: str = 'relu',
            output_activation: str = 'softmax'
        ):
            super().__init__()
            
            # Build switched linear problem
            n_phases = 50
            _, _, cost_function, gradient_function = self.switched_problem(n_phases)
            
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

            time_horizon = 10

            x0 = np.array([2, -1, 5])
            
            xr = np.array([1, -3])
            
            swi_lin = SwiLin(
                n_phases, 
                n_states,
                n_inputs,
                time_horizon, 
                auto=False, 
            )
            
            # Load model
            swi_lin.load_model(model)

            Q = 10. * np.eye(n_states)
            R = 10. * np.eye(n_inputs)
            E = 1. * np.eye(n_states)

            swi_lin.precompute_matrices(x0, Q, R, E)
            x0 = np.append(x0, 1)  # augment with 1 for affine term
            J_func = swi_lin.cost_function(R, x0)
                
            grad_J_u = []
            grad_J_delta = []

            for k in range(n_phases):
                # Compute gradient of the cost
                du, d_delta = swi_lin.grad_cost_function(k, R)
                # print(f"Length du: {len(du)}")

                grad_J_u += du
                grad_J_delta.append(d_delta)

            grad_J = ca.vertcat(*grad_J_delta, *grad_J_u)

            # keep the original stacked forms if needed
            grad_J_u = np.hstack(grad_J_u)
            grad_J_delta = np.hstack(grad_J_delta)
            
            # Create a CasADi function for the gradient
            grad_J_func = ca.Function('grad_J', [*swi_lin.u, *swi_lin.delta], [grad_J])
        
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
        y_train: torch.Tensor,
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
        y_train = y_train.to(device)
        
        if X_val is not None:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
        
        n_samples = X_train.shape[0]
        
        # Loss function
        if network.output_activation == 'softmax':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Gradient function
        def gradient_func(params, indices=None, data=None):
            network.set_flat_params(params)
            network.zero_grad()
            
            if indices is not None:
                X_batch = X_train[indices]
                y_batch = y_train[indices]
            else:
                X_batch = X_train
                y_batch = y_train
            
            output = network(X_batch)
            
            # Compute Jacobian: derivative of each output w.r.t. parameters
            # Sum the output over the batch dimension to get a scalar loss
            jacobian = []
            for i in range(output.shape[1] if output.dim() > 1 else 1):
                network.zero_grad()
                if output.dim() > 1:
                    output[:, i].sum().backward(retain_graph=True)
                else:
                    output.sum().backward(retain_graph=True)
                
                grads_i = []
                for param in network.parameters():
                    if param.grad is not None:
                        grads_i.append(param.grad.view(-1).clone())
                jacobian.append(torch.cat(grads_i))

            jacobian = torch.stack(jacobian)  # Shape: (n_outputs, n_params)
            
            # Include the derivative of the loss w.r.t. the outputs of the NN
            
            if network.output_activation == 'softmax' and y_batch.dtype == torch.long:
                loss = criterion(output, y_batch)
            else:
                loss = criterion(output, y_batch)
            
            print(f"Loss: {loss.item()}")
            print(f"Loss shape: {loss.shape}")
            input("Press Enter to continue...")
            loss.backward()
            
            # Collect gradients
            grads = []
            for param in network.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
    
            pippo = torch.cat(grads)
            print(f"Gradient shape: {pippo.shape}")
            input("Press Enter to continue...")
            return torch.cat(grads)
        
        # Loss function
        def loss_func(params, data=None):
            network.set_flat_params(params)
            output = network(X_train)
            
            if network.output_activation == 'softmax' and y_train.dtype == torch.long:
                loss = criterion(output, y_train)
            else:
                loss = criterion(output, y_train)
            
            return loss
        
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
        
        # Compute final accuracy
        if verbose:
            with torch.no_grad():
                train_pred = network(X_train)
                train_acc = compute_accuracy_torch(train_pred, y_train)
                print(f"\nFinal Training Accuracy: {train_acc:.4f}")
                
                if X_val is not None and y_val is not None:
                    val_pred = network(X_val)
                    val_acc = compute_accuracy_torch(val_pred, y_val)
                    print(f"Final Validation Accuracy: {val_acc:.4f}")
        
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


def example_mnist_torch():
    """
    Example: Train neural network on MNIST-like data using PyTorch optimizer.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping GPU example.")
        return
    
    print("=" * 70)
    print("Example: Neural Network Training")
    print("=" * 70)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 10000
    n_features = 28 * 28
    n_classes = 10
    
    X_train = torch.randn(n_samples, n_features)
    y_train = torch.randint(0, n_classes, (n_samples,))
    
    X_val = torch.randn(2000, n_features)
    y_val = torch.randint(0, n_classes, (2000,))
    
    # Create network
    network = SwiLinNN(
        layer_sizes=[n_features, 128, 256, n_classes],
        activation='relu',
        # output_activation='softmax'
    )
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    params_opt, history = train_neural_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
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
    example_mnist_torch()
    end = time.time()
    print(f"PyTorch example took {end - start:.2f} seconds")
