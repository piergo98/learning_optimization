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
import scipy.io
from scipy.linalg import solve_continuous_are
from typing import Optional, Callable, Tuple, Dict, List
import warnings
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU training will not be available.")
    
from switched_linear_torch import SwiLin


# ============================================================================
# PyTorch Implementation
# ============================================================================

if TORCH_AVAILABLE:
    from torch.utils.tensorboard import SummaryWriter
    
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
            n_phases: int = 80,
            activation: str = 'relu',
            output_activation: str = 'linear'
        ):
            super().__init__()
            
            # Build switched linear problem
            self.n_phases = n_phases
            self.sys = self.switched_problem(self.n_phases)
            
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

            self.time_horizon = 2.0
            
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

            swi_lin.load_weights(Q, R, E)
            
            # Store the cost function (now returns a callable)
            # self.cost_func = swi_lin.cost_function(R, sym_x0=True)
            # self.R = R
            
            return swi_lin
        
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

        def save(self, path: str) -> None:
            """Save the network parameters (state_dict) to `path`.

            Args:
                path: filesystem path where to save the state_dict (e.g. 'model.pt').
            """
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)

        def load(self, path: str, map_location: Optional[str] = None) -> None:
            """Load the network parameters from `path` into this model.

            Args:
                path: filesystem path from which to load the state_dict.
                map_location: optional torch.load map_location argument.
            """
            map_loc = map_location if map_location is not None else None
            state = torch.load(path, map_location=map_loc)
            self.load_state_dict(state)
    
    def evaluate_cost_functional(swi: SwiLin, u_all: torch.Tensor, delta_all: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Pure functional evaluation of the LQR-style cost.
        Args:
        swi: SwiLin instance (used only for A,B,Q,R and helper functions)
        u_all: tensor shape (n_phases, n_inputs)
        delta_all: tensor shape (n_phases,) or (n_phases,1)
        x0: tensor shape (n_states,) or (n_states,1)
        Returns:
        J: scalar tensor (dtype/device consistent with inputs)
        """
        device = u_all.device if torch.is_tensor(u_all) else swi.device
        dtype = u_all.dtype if torch.is_tensor(u_all) else swi.dtype

        n_ph = swi.n_phases
        n_x = swi.n_states
        n_u = swi.n_inputs

        # ensure shapes
        u_all = u_all.to(device=device, dtype=dtype)
        delta_all = delta_all.view(n_ph).to(device=device, dtype=dtype)
        x = x0.to(device=device, dtype=dtype).reshape(n_x, 1)  # column

        # Local container for Ei, phi_f_i, Li, Mi, Ri, Hi if needed
        Es = []
        phi_fs = []
        Lis = []
        Mis = []
        Ris = []

        for i in range(n_ph):
            u_i = u_all[i].reshape(-1) if n_u > 0 else None
            delta_i = delta_all[i]

            # mat_exp_prop returns different tuples depending on auto; adapt:
            out = swi.mat_exp_prop(i, u_i, delta_i)
            # for 'exp' non-autonomous our implementation returns (Ei, phi_f_i, Hi, Li, Mi, Ri)
            if not swi.auto:
                Ei, phi_f_i, Hi, Li, Mi, Ri = out
                # phi_f_i returned as column vector (n_x x 1) in the file; ensure shape
                phi_f_i = phi_f_i.reshape(n_x, 1)
            else:
                Ei, Li = out
                phi_f_i = torch.zeros((n_x, 1), device=device, dtype=dtype)
                Mi = torch.zeros((n_x, n_u), device=device, dtype=dtype) if n_u > 0 else None
                Ri = torch.zeros((n_u, n_u), device=device, dtype=dtype) if n_u > 0 else None

            Es.append(Ei)
            phi_fs.append(phi_f_i)
            Lis.append(Li)
            Mis.append(Mi)
            Ris.append(Ri)

        # Backward pass to compute S_0 (local, no mutation):
        # Terminal cost augmentation
        Eterm = swi.E_term.to(device=device, dtype=dtype)
        S_prev = 0.5 * torch.block_diag(Eterm, torch.zeros(1, 1, device=device, dtype=dtype))

        for i in range(n_ph - 1, -1, -1):
            Ei = Es[i]
            phi_f_i = phi_fs[i]
            Li = Lis[i]
            Mi = Mis[i]
            Ri = Ris[i]

            # build S_int like in _S_matrix_exp
            S_int = torch.zeros((n_x + 1, n_x + 1), device=device, dtype=dtype)
            S_int[:n_x, :n_x] = Li
            if n_u > 0:
                ui_col = u_all[i].reshape(-1, 1)
                Mi_ui = Mi @ ui_col
                S_int[:n_x, n_x:] = Mi_ui
                S_int[n_x:, :n_x] = Mi_ui.T
                S_int[n_x:, n_x:] = ui_col.T @ Ri @ ui_col

            # transition matrix phi (n_x+1 x n_x+1)
            phi_i = swi.transition_matrix(Ei, phi_f_i)

            S_curr = 0.5 * S_int + (phi_i.T @ S_prev @ phi_i)
            S_prev = S_curr

        S0 = S_prev  # S at time 0

        # Forward propagate states locally (to compute state-dependent term)
        x_curr = x
        for i in range(n_ph):
            Ei = Es[i]
            phi_f_i = phi_fs[i]
            if n_u > 0:
                ui_col = u_all[i].reshape(-1, 1)
                x_next = Ei @ x_curr + (phi_f_i)  # phi_f_i already includes multiplication by u in mat_exp_prop
            else:
                x_next = Ei @ x_curr
            x_curr = x_next

        x0_aug = torch.cat([x.reshape(-1, 1), torch.ones(1, 1, device=device, dtype=dtype)], dim=0)

        # Control cost G
        if n_u > 0:
            # per-phase u^T R u * delta
            G_terms = []
            Rmat = swi.R.to(device=device, dtype=dtype)
            for i in range(n_ph):
                ui = u_all[i].reshape(-1, 1)
                G_terms.append(0.5 * (ui.T @ Rmat @ ui) * delta_all[i])
            G0 = sum(G_terms)
        else:
            G0 = torch.tensor(0.0, device=device, dtype=dtype)

        J = 0.5 * (x0_aug.T @ S0 @ x0_aug) + G0
        # Return scalar tensor
        return J.squeeze()
    
    def evaluate_cost_functional_batch(
        swi: SwiLin,
        u_all_batch: torch.Tensor,
        delta_all_batch: torch.Tensor,
        x0_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized evaluation of the LQR-style cost over a batch.

        Args:
            swi: SwiLin instance (used for model matrices and helpers)
            u_all_batch: tensor shape (B, n_phases, n_inputs)
            delta_all_batch: tensor shape (B, n_phases)
            x0_batch: tensor shape (B, n_states)

        Returns:
            J_batch: tensor shape (B,) with per-sample costs
        """
        device = u_all_batch.device if torch.is_tensor(u_all_batch) else swi.device
        dtype = u_all_batch.dtype if torch.is_tensor(u_all_batch) else swi.dtype

        B = u_all_batch.shape[0]
        n_ph = swi.n_phases
        n_x = swi.n_states
        n_u = swi.n_inputs

        # Ensure tensors on correct device/dtype
        u_all_batch = u_all_batch.to(device=device, dtype=dtype)
        delta_all_batch = delta_all_batch.to(device=device, dtype=dtype).view(B, n_ph)
        x0_batch = x0_batch.to(device=device, dtype=dtype).view(B, n_x)

        # Containers per phase (each element will be batch-shaped)
        Es = [None] * n_ph
        phi_fs = [None] * n_ph
        Lis = [None] * n_ph
        Mis = [None] * n_ph
        Ris = [None] * n_ph

        # Useful constants
        Q = swi.Q.to(dtype=dtype, device=device)
        R = swi.R.to(dtype=dtype, device=device) if n_u > 0 else None
        Eterm = swi.E_term.to(dtype=dtype, device=device)

        # For each phase compute batched matrices
        for i in range(n_ph):
            A = swi.A[i].to(dtype=dtype, device=device)
            Bmat = swi.B[i].to(dtype=dtype, device=device) if n_u > 0 else None

            # Build big C matrix once (same across batch) as in _mat_exp_prop_exp
            if not swi.auto:
                m = n_u
                Mdim = 3 * n_x + m
                C_base = torch.zeros((Mdim, Mdim), dtype=dtype, device=device)
                C_base[:n_x, :n_x] = -A.T
                C_base[:n_x, n_x:2*n_x] = torch.eye(n_x, dtype=dtype, device=device)
                C_base[n_x:2*n_x, n_x:2*n_x] = -A.T
                C_base[n_x:2*n_x, 2*n_x:3*n_x] = Q
                C_base[2*n_x:3*n_x, 2*n_x:3*n_x] = A
                C_base[2*n_x:3*n_x, 3*n_x:] = Bmat

                # Create batch of C scaled by delta
                deltas_i = delta_all_batch[:, i].view(B, 1, 1)
                C_batch = C_base.unsqueeze(0) * deltas_i

                # Batched matrix exponential
                exp_C = torch.linalg.matrix_exp(C_batch)

                # Extract pieces
                F3 = exp_C[:, 2*n_x:3*n_x, 2*n_x:3*n_x]  # (B, n_x, n_x)
                G2 = exp_C[:, n_x:2*n_x, 2*n_x:3*n_x]  # (B, n_x, n_x)
                G3 = exp_C[:, 2*n_x:3*n_x, 3*n_x:]      # (B, n_x, m)
                H2 = exp_C[:, n_x:2*n_x, 3*n_x:]       # (B, n_x, m)
                K1 = exp_C[:, :n_x, 3*n_x:]            # (B, n_x, m)

                Ei_batch = F3
                Li_batch = torch.matmul(F3.transpose(-1, -2), G2)

                # phi_f_i = phi_f_i_ @ ui for each sample
                ui_batch = u_all_batch[:, i, :].view(B, n_u, 1) if n_u > 0 else None
                if n_u > 0:
                    phi_f_i_ = G3  # (B, n_x, m)
                    # phi_f: (B, n_x, 1)
                    phi_f_batch = torch.matmul(phi_f_i_, ui_batch)

                    # Mi = F3.T @ H2 -> (B, n_x, m)
                    Mi_batch = torch.matmul(F3.transpose(-1, -2), H2)

                    # Ri: temp = B.T @ F3.T @ K1  -> (B, m, m)
                    # compute F3.T @ K1 -> (B, n_x, m)
                    tmp = torch.matmul(F3.transpose(-1, -2), K1)
                    # Bmat.T (m,n_x) @ tmp (B, n_x, m) -> (B, m, m)
                    temp = torch.matmul(Bmat.T.unsqueeze(0), tmp)
                    Ri_batch = temp + temp.transpose(-1, -2)
                else:
                    phi_f_batch = torch.zeros((B, n_x, 1), device=device, dtype=dtype)
                    Mi_batch = torch.zeros((B, n_x, 0), device=device, dtype=dtype)
                    Ri_batch = torch.zeros((B, 0, 0), device=device, dtype=dtype)

                Es[i] = Ei_batch
                phi_fs[i] = phi_f_batch
                Lis[i] = Li_batch
                Mis[i] = Mi_batch
                Ris[i] = Ri_batch
            else:
                # Autonomous case: simpler (Ei depends only on delta)
                deltas_i = delta_all_batch[:, i].view(B, 1, 1)
                Ei_batch = torch.linalg.matrix_exp(A.unsqueeze(0) * deltas_i)
                Li_batch = torch.zeros((B, n_x, n_x), device=device, dtype=dtype)
                Es[i] = Ei_batch
                phi_fs[i] = torch.zeros((B, n_x, 1), device=device, dtype=dtype)
                Lis[i] = Li_batch
                Mis[i] = torch.zeros((B, n_x, 0), device=device, dtype=dtype)
                Ris[i] = torch.zeros((B, 0, 0), device=device, dtype=dtype)

        # Backward recursion to compute S0 per sample
        # Initialize S_prev as (B, n_x+1, n_x+1)
        E_aug = torch.zeros((n_x+1, n_x+1), device=device, dtype=dtype)
        E_aug[:n_x, :n_x] = Eterm
        S_prev = 0.5 * E_aug.unsqueeze(0).expand(B, n_x+1, n_x+1).clone()

        for i in range(n_ph-1, -1, -1):
            Ei_b = Es[i]
            phi_f_b = phi_fs[i]
            Li_b = Lis[i]
            Mi_b = Mis[i]
            Ri_b = Ris[i]

            # Build S_int batch
            S_int = torch.zeros((B, n_x+1, n_x+1), device=device, dtype=dtype)
            S_int[:, :n_x, :n_x] = Li_b

            if n_u > 0:
                ui_col = u_all_batch[:, i, :].view(B, n_u, 1)
                # Mi_b: (B, n_x, n_u) -> Mi_ui: (B, n_x, 1)
                Mi_ui = torch.matmul(Mi_b, ui_col)
                S_int[:, :n_x, n_x:] = Mi_ui
                S_int[:, n_x:, :n_x] = Mi_ui.transpose(-1, -2)
                # scalar term: ui^T Ri ui -> (B,1,1)
                tmp = torch.matmul(Ri_b, ui_col)  # (B, n_u, 1)
                uiRiui = torch.matmul(ui_col.transpose(-1, -2), tmp)  # (B,1,1)
                S_int[:, n_x:, n_x:] = uiRiui

            # Build phi batch (B, n_x+1, n_x+1)
            phi = torch.zeros((B, n_x+1, n_x+1), device=device, dtype=dtype)
            phi[:, :n_x, :n_x] = Ei_b
            phi[:, :n_x, n_x:n_x+1] = phi_f_b
            phi[:, -1, -1] = 1.0

            # S_curr = 0.5*S_int + phi^T * S_prev * phi
            S_curr = 0.5 * S_int + torch.matmul(phi.transpose(-1, -2), torch.matmul(S_prev, phi))
            S_prev = S_curr

        S0_batch = S_prev

        # Forward propagate states for each sample
        x_curr = x0_batch.view(B, n_x, 1)
        for i in range(n_ph):
            Ei_b = Es[i]
            phi_f_b = phi_fs[i]
            if n_u > 0:
                # Ei_b: (B,n_x,n_x), x_curr: (B,n_x,1) -> (B,n_x,1)
                x_next = torch.matmul(Ei_b, x_curr) + phi_f_b
            else:
                x_next = torch.matmul(Ei_b, x_curr)
            x_curr = x_next

        # Augment x0 for bilinear form
        x0_aug = torch.cat([x0_batch.view(B, n_x, 1), torch.ones((B, 1, 1), device=device, dtype=dtype)], dim=1)

        # Compute quadratic term: 0.5 * x0_aug^T * S0 * x0_aug -> (B,1,1)
        quad = torch.matmul(x0_aug.transpose(-1, -2), torch.matmul(S0_batch, x0_aug)).squeeze(-1).squeeze(-1)

        # Compute G per sample
        if n_u > 0:
            # u_all_batch: (B, n_ph, n_u)
            per_phase_terms = []
            for i in range(n_ph):
                u_b = u_all_batch[:, i, :]  # (B, n_u)
                # (B, n_u) @ (n_u,n_u) -> (B, n_u)
                uR = torch.matmul(u_b, R)
                per = (uR * u_b).sum(dim=1)  # (B,)
                per_phase_terms.append(0.5 * per * delta_all_batch[:, i])

            G0 = torch.stack(per_phase_terms, dim=1).sum(dim=1)  # (B,)
        else:
            G0 = torch.zeros(B, device=device, dtype=dtype)

        J_batch = 0.5 * quad + G0
        return J_batch
    
    def train_neural_network(
        network: SwiLinNN,
        X_train: torch.Tensor,
        y_train: Optional[torch.Tensor] = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        batch_size: int = 32,
        device: str = 'cpu',
        # Resampling options: regenerate new random samples every N epochs
        resample_every: Optional[int] = None,
        resample_fn: Optional[Callable[[int], torch.Tensor]] = None,
        resample_val: bool = False,
        verbose: bool = True,
        tensorboard_logdir: Optional[str] = None,
        log_histograms: bool = False,
        save_history: bool = False,
        save_history_path: Optional[str] = None,
        save_model: bool = False,
        save_model_path: Optional[str] = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-6,
        early_stopping_monitor: str = 'val_loss',
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
        weight_decay : float
            L2 regularization weight decay.
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        device : str
            Device: 'cpu' or 'cuda'.
        resample_every : int or None
            If provided, regenerate training samples every `resample_every` epochs.
            Set to `None` to disable automatic resampling.
        resample_fn : callable or None
            Function called to generate new samples. It should accept the current
            epoch (int) and return either a new `X_train` tensor or a tuple
            `(X_train, X_val)`. If `None` a default uniform sampler between the
            observed min/max of `X_train` is used.
        resample_val : bool
            If True and `resample_fn` returns validation samples, replace `X_val`
            as well when resampling.
        verbose : bool
            Print training progress.
        tensorboard_logdir : str, optional
            Directory for TensorBoard logs.
        log_histograms : bool
            Whether to log parameter histograms to TensorBoard.
        save_history : bool
            Whether to save training history to JSON.
        save_history_path : str, optional
            Path to save training history JSON file.
        save_model : bool
            Whether to save the trained model.
        save_model_path : str, optional
            Path to save the trained model.
        early_stopping : bool
            Whether to use early stopping.
        early_stopping_patience : int
            Number of epochs with no improvement after which training will be stopped.
        early_stopping_min_delta : float
            Minimum change in monitored quantity to qualify as an improvement.
        early_stopping_monitor : str
            Metric to monitor for early stopping: 'val_loss' or 'train_loss'.
            
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

        # Setup a default resampling function if requested but none provided.
        # Default resampler draws uniformly between observed min/max of X_train
        if resample_every is not None and resample_every > 0 and resample_fn is None:
            try:
                # x_min = float(X_train.min().item())
                x_min = -5.0
                # x_max = float(X_train.max().item())
                x_max = 5.0
            except Exception:
                x_min, x_max = -1.0, 1.0

            def _default_resample_fn(epoch, shape=X_train.shape, dtype=X_train.dtype, device_str=device, xmin=x_min, xmax=x_max):
                # create tensor on correct device/dtype
                dev = device_str
                out = torch.empty(shape, dtype=dtype, device=dev).uniform_(xmin, xmax)
                return out

            resample_fn = _default_resample_fn
        
        n_samples = X_train.shape[0]
        n_inputs = network.sys.n_inputs
        
        # Initialize PyTorch optimizer
        if optimizer.lower() == 'adam':
            torch_optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            torch_optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'rmsprop':
            torch_optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Supported: 'adam', 'sgd', 'rmsprop'")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None,
            'epochs': []
        }
        
        # Early stopping setup
        if early_stopping:
            if early_stopping_monitor == 'val_loss' and X_val is None:
                warnings.warn("Early stopping monitor is 'val_loss' but no validation data provided. Switching to 'train_loss'.")
                early_stopping_monitor = 'train_loss'
            
            best_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            best_model_state = None
            
            if verbose:
                print(f"Early stopping enabled: monitoring '{early_stopping_monitor}' with patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        
        # Setup TensorBoard writer if requested
        writer = SummaryWriter(log_dir=tensorboard_logdir) if tensorboard_logdir is not None else None

        # Determine history save path
        if save_history:
            if save_history_path is None:
                if tensorboard_logdir is not None:
                    save_history_path = os.path.join(tensorboard_logdir, 'history.json')
                else:
                    save_history_path = os.path.join(os.getcwd(), 'training_history.json')


        # Training loop
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Optionally resample training (and validation) data every `resample_every` epochs
            if resample_every is not None and resample_every > 0 and epoch > 0 and (epoch % resample_every) == 0:
                if resample_fn is None:
                    warnings.warn("resample_every set but resample_fn is None; skipping resampling.")
                else:
                    try:
                        new_data = resample_fn(epoch)
                        # support returning either X_train or (X_train, X_val)
                        if isinstance(new_data, (list, tuple)) and len(new_data) == 2:
                            new_X_train, new_X_val = new_data
                        else:
                            new_X_train, new_X_val = new_data, None

                        if not torch.is_tensor(new_X_train):
                            new_X_train = torch.as_tensor(new_X_train)
                        X_train = new_X_train.to(device)
                        n_samples = X_train.shape[0]

                        if resample_val and new_X_val is not None:
                            if not torch.is_tensor(new_X_val):
                                new_X_val = torch.as_tensor(new_X_val)
                            X_val = new_X_val.to(device)

                        if verbose:
                            print(f"Resampled training data at epoch {epoch + 1}")
                    except Exception as e:
                        warnings.warn(f"Resampling failed at epoch {epoch + 1}: {e}")

            # Create random batches
            indices = torch.randperm(n_samples, device=device)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                current_batch_size = X_batch.shape[0]
                
                # Zero gradients
                torch_optimizer.zero_grad()
                
                # Forward pass
                output = network(X_batch)
                
                # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
                T_tensor = torch.tensor(network.sys.time_horizon, device=output.device, dtype=output.dtype)

                # Handle batch dimension properly
                n_control_outputs = network.n_phases * n_inputs
                controls = output[:, :n_control_outputs] # shape (batch_size, n_phases * n_inputs)
                delta_raw = output[:, n_control_outputs:]
                
                # Apply softmax and scale deltas
                delta_normalized = F.softmax(delta_raw, dim=-1)
                deltas = delta_normalized * T_tensor # shape (batch_size, n_phases)
                
                # Clip controls using tanh-based soft clipping to preserve gradients
                u_min = -1.0  # Define your lower bound
                u_max = 1.0   # Define your upper bound
                u_center = (u_max + u_min) / 2.0
                u_range = (u_max - u_min) / 2.0
                # Soft clipping: maps (-inf, inf) to (u_min, u_max) smoothly
                controls = u_center + u_range * torch.tanh(controls)
                
                transformed_output = torch.cat([controls, deltas], dim=-1) # shape (batch_size, n_phases * (n_inputs + 1))
                
                # Instead of the for loop, I have to give the full batch to the cost function

                # Vectorized batch loss computation
                # reshape controls to (B, n_phases, n_inputs)
                B_batch = current_batch_size
                controls_reshaped = controls.view(B_batch, network.n_phases, n_inputs)
                deltas_batch = deltas.view(B_batch, network.n_phases)
                x0_batch = X_batch

                J_batch = evaluate_cost_functional_batch(network.sys, controls_reshaped, deltas_batch, x0_batch)
                loss = J_batch.mean()
                
                # Backward pass
                loss.backward()
                # Compute gradient norm for logging
                grad_norm = None
                if writer is not None:
                    tot = torch.tensor(0.0, device=device)
                    for p in network.parameters():
                        if p.grad is not None:
                            tot = tot + p.grad.detach().to(device).pow(2).sum()
                    grad_norm = torch.sqrt(tot).item()

                # Optimizer step
                torch_optimizer.step()

                # Log per-batch stats to TensorBoard (optional)
                if writer is not None:
                    global_step = epoch * max(1, n_samples // batch_size) + n_batches
                    writer.add_scalar('train/batch_loss', loss.item(), global_step)
                    if grad_norm is not None:
                        writer.add_scalar('train/batch_grad_norm', grad_norm, global_step)
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss for the epoch
            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            history['epochs'].append(epoch)
            
            # Validation loss
            if X_val is not None:
                with torch.no_grad():
                    val_output = network(X_val)
                    
                    # Transform validation output
                    n_control_outputs = network.n_phases * n_inputs
                    val_controls = val_output[:, :n_control_outputs]
                    # Clip controls using tanh-based soft clipping to preserve gradients
                    u_min = -1.0  # Define your lower bound
                    u_max = 1.0   # Define your upper bound
                    u_center = (u_max + u_min) / 2.0
                    u_range = (u_max - u_min) / 2.0
                    # Soft clipping: maps (-inf, inf) to (u_min, u_max) smoothly
                    val_controls = u_center + u_range * torch.tanh(val_controls)
                    val_delta_raw = val_output[:, n_control_outputs:]
                    val_delta_normalized = F.softmax(val_delta_raw, dim=-1)
                    val_deltas = val_delta_normalized * T_tensor
                    val_transformed = torch.cat([val_controls, val_deltas], dim=-1)
                    
                    # Vectorized validation loss
                    Bv = X_val.shape[0]
                    val_controls = val_controls.view(Bv, network.n_phases, n_inputs)
                    val_deltas = val_deltas.view(Bv, network.n_phases)
                    J_val = evaluate_cost_functional_batch(network.sys, val_controls, val_deltas, X_val)
                    avg_val_loss = J_val.mean().item()
                    history['val_loss'].append(avg_val_loss)
            
            # Step the learning rate scheduler
            if X_val is not None:
                scheduler.step(avg_val_loss)
            else:
                scheduler.step(avg_train_loss)

            # Write epoch-level scalars to TensorBoard
            if writer is not None:
                writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
                writer.add_scalar('train/learning_rate', torch_optimizer.param_groups[0]['lr'], epoch)
                if X_val is not None:
                    writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
                # Optionally log parameter histograms once per epoch
                if log_histograms:
                    for name, param in network.named_parameters():
                        writer.add_histogram(f'params/{name}', param.detach().cpu().numpy(), epoch)

            # Save history to disk each epoch if requested
            if save_history:
                try:
                    serial = {}
                    for k, v in history.items():
                        if v is None:
                            serial[k] = None
                        elif isinstance(v, list):
                            serial[k] = [float(x) for x in v]
                        else:
                            serial[k] = v
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
                    with open(save_history_path, 'w') as fh:
                        json.dump(serial, fh, indent=2)
                except Exception:
                    # Don't interrupt training on save failure; warn instead
                    warnings.warn(f"Failed to save training history to {save_history_path}")
            
            # Print progress
            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}")
            
            # Early stopping check
            if early_stopping:
                # Determine which loss to monitor
                current_loss = avg_val_loss if early_stopping_monitor == 'val_loss' else avg_train_loss
                
                # Check if there's improvement
                if current_loss < best_loss - early_stopping_min_delta:
                    best_loss = current_loss
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
                    if verbose and epoch > 0:
                        print(f"  → New best {early_stopping_monitor}: {best_loss:.6f}")
                else:
                    patience_counter += 1
                    if verbose and patience_counter > 0 and (epoch + 1) % max(1, n_epochs // 10) == 0:
                        print(f"  → No improvement for {patience_counter} epoch(s)")
                
                # Check if we should stop
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        print(f"Best {early_stopping_monitor}: {best_loss:.6f} at epoch {best_epoch + 1}")
                    
                    # Restore best model state
                    if best_model_state is not None:
                        network.load_state_dict(best_model_state)
                        if verbose:
                            print("Restored best model weights")
                    
                    break
        
        # Get final parameters
        params_optimized = network.get_flat_params()
        
        # Optionally save the trained model parameters
        if save_model:
            if save_model_path is None:
                if tensorboard_logdir is not None:
                    save_model_path = os.path.join(tensorboard_logdir, 'model_state_dict.pt')
                else:
                    save_model_path = os.path.join(os.getcwd(), 'model_state_dict.pt')
            try:
                network.save(save_model_path)
                if verbose:
                    print(f"Saved model state_dict to: {save_model_path}")
            except Exception:
                warnings.warn(f"Failed to save model to {save_model_path}")

        # Add early stopping info to history
        if early_stopping:
            history['early_stopping'] = {
                'triggered': patience_counter >= early_stopping_patience,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'monitored_metric': early_stopping_monitor,
                'patience': early_stopping_patience,
                'final_epoch': epoch
            }

        # Print final losses
        if verbose:
            print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
            if X_val is not None and history['val_loss']:
                print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
            if early_stopping and history.get('early_stopping', {}).get('triggered', False):
                print(f"\nEarly stopping was triggered:")
                print(f"  Best {early_stopping_monitor}: {best_loss:.6f} at epoch {best_epoch + 1}")
                print(f"  Training stopped at epoch {epoch + 1}")

        return params_optimized, history
    

# ============================================================================
# Example Usage
# ============================================================================


def example_torch():
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
    n_samples_train = 10000
    n_samples_val = 200
    n_phases = 50
    n_control_inputs = 2
    n_NN_inputs = 3
    n_NN_outputs = n_phases * (n_control_inputs + 1)
    
    X_train = torch.empty(n_samples_train, n_NN_inputs).uniform_(-10.0, 10.0)
    
    
    X_val = torch.empty(n_samples_val, n_NN_inputs).uniform_(-10.0, 10.0)
    
    # Create network
    network = SwiLinNN(
        layer_sizes=[n_NN_inputs, 50, 50, n_NN_outputs],
        n_phases=n_phases,
        activation='relu',
        output_activation='linear'
    )
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Store the path where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    date = subprocess.check_output(['date', '+%Y%m%d_%H%M%S']).decode('utf-8').strip()
    tensorboard_logdir = os.path.join(script_dir, "..", "logs", date)
    model_name = f"nahs_torch_{date}.pt"
    models_dir = os.path.join(script_dir, "..", "models", model_name)
    
    params_opt, history = train_neural_network(
        network=network,
        X_train=X_train,
        # y_train=None,
        X_val=X_val,
        # y_val=None,
        optimizer='adam',
        learning_rate=0.001,
        weight_decay=1e-4,
        n_epochs=400,
        resample_every=None,
        resample_fn=None,
        resample_val=False,
        early_stopping=True,
        early_stopping_patience=30,
        early_stopping_min_delta=1e-4,
        batch_size=n_samples_train,
        device=device,
        verbose=False,
        tensorboard_logdir=tensorboard_logdir,
        log_histograms=False,
        save_model=True,
        save_model_path=models_dir
        
    )
    
    print("\nTraining complete!")
    
def load_data(filename):
    """
    Load data from file <filename>, which has to be in the data folder.
    The function loads both csv or mat files
    """
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(data_folder, filename)

    if filename.endswith('.csv'):
        loaded_data = np.loadtxt(file_path, delimiter=',')
    elif filename.endswith('.mat'):
        loaded_data = scipy.io.loadmat(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .mat files.")
    
    # Handle data after loading
    keys_to_keep = ['n_phases', 'controls', 'phases_duration']

    data = {k: loaded_data[k] for k in keys_to_keep}
    # Normalize and reshape controls into shape (n_inputs, n_phases)
    controls = np.asarray(data['controls']).ravel()
    n_phases = int(np.squeeze(np.asarray(data['n_phases'])))
    if controls.size % n_phases != 0:
        raise ValueError(f"Controls length ({controls.size}) is not divisible by n_phases ({n_phases}).")
    n_inputs = controls.size // n_phases
    controls = controls.reshape((n_inputs, n_phases))
    data['n_inputs'] = n_inputs

    # Ensure phases_duration is a 1D array
    data['phases_duration'] = np.asarray(data['phases_duration']).ravel()

    return data


if __name__ == "__main__":
    import time
    
    # print("\n" + "=" * 70 + "\n")
    # data_file = 'optimal_params.mat'
    # data = load_data(data_file)
    
    # Run PyTorch example
    start = time.time()
    example_torch()
    end = time.time()
    print(f"PyTorch example took {end - start:.2f} seconds")
