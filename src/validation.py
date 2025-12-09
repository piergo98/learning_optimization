"""
This file provides a set of validation methods for the trained machine learning model.
"""

import numpy as np
import os, subprocess, sys
import scipy.io
from scipy.linalg import solve_continuous_are
from typing import Optional, Callable, Tuple, Dict, List
import warnings
import json
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU training will not be available.")
    
from ocslc.switched_linear import SwiLin as SwiLin_casadi
from switched_linear_torch import SwiLin
from training import SwiLinNN


class ModelValidator:
    def __init__(self, model: SwiLinNN, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelValidator.")
        
        # Initialize the validator with the model and device
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_data(self, filename: str) -> None:
        """
        Load data from file <filename>, which has to be in the data folder.
        The function loads both csv or mat files
        
        Args:
            filename (str): Name of the file to load data from.
            
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
        controls = torch.as_tensor(data['controls'], dtype=torch.float32).reshape(-1)
        n_phases = int(torch.as_tensor(data['n_phases']).squeeze().item())
        if controls.numel() % n_phases != 0:
            raise ValueError(f"Controls length ({controls.numel()}) is not divisible by n_phases ({n_phases}).")
        n_inputs = controls.numel() // n_phases
        controls = controls.reshape(n_inputs, n_phases)
        self.controls_gt = controls
        self.n_inputs = int(n_inputs)

        # Ensure phases_duration is a 1D tensor
        self.phases_duration_gt = torch.as_tensor(data['phases_duration'], dtype=torch.float32).reshape(-1)
        self.n_phases = n_phases

    def validate_on_data(self, x0, criterion: Callable) -> Dict[str, float]:
        """
        Validate the model on a dataset loaded from a file.
        
        Args:
            x0: Initial state tensor.
            criterion (Callable): Loss function to evaluate the model.
            
        Returns:
            dict: A dictionary containing validation metrics.
        """
        # Load data
        # self.load_data(filename)
        n_inputs = self.n_inputs
        n_phases = self.n_phases
        controls = self.controls_gt.to(self.device)
        phases_duration = self.phases_duration_gt.to(self.device)
        
        with torch.no_grad():
            x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

            prediction = self.model(x0)
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(self.model.sys.time_horizon, device=prediction.device, dtype=prediction.dtype)

            # Handle batch dimension properly
            n_control_outputs = self.model.n_phases * n_inputs
            pred_u = prediction[:, :n_control_outputs] # shape (batch_size, n_phases * n_inputs)
            pred_delta_raw = prediction[:, n_control_outputs:]
            
            # Apply softmax and scale deltas
            delta_normalized = F.softmax(pred_delta_raw, dim=-1)
            pred_deltas = delta_normalized * T_tensor # shape (batch_size, n_phases)
            
            # Clip controls using tanh-based soft clipping to preserve gradients
            u_min = -1.0  # Define your lower bound
            u_max = 1.0   # Define your upper bound
            u_center = (u_max + u_min) / 2.0
            u_range = (u_max - u_min) / 2.0
            # Soft clipping: maps (-inf, inf) to (u_min, u_max) smoothly
            pred_u = u_center + u_range * torch.tanh(pred_u)
            
            # Compute the loss
            loss = criterion(pred_u, controls.view_as(pred_u)) + criterion(pred_deltas, phases_duration.view_as(pred_deltas))
        
        return loss.item()

    def validate_on_custom_function(self, x0) -> Dict[str, float]:
        """
        Validate the cost function value based on the output of the model.

        """
        # First compute the cost function value using the Casadi SwiLin system
        # Create a simple switched linear system
        n_phases = 80
        n_states = 3
        n_inputs = 1
        time_horizon = 10.0

        model = {
                'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
                'B': [np.array([[0.25], [2], [0]])],
            }

        # print("Creating SwiLin system...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        swi_lin = SwiLin(n_phases, n_states, n_inputs, time_horizon, auto=False, device=device) 
        swi_lin_casadi = SwiLin_casadi(n_phases, n_states, n_inputs, time_horizon, auto=False)
        swi_lin.load_model(model)
        swi_lin_casadi.load_model(model)

        # print("Precomputing matrices...")
        # Convert x0 to numpy for CasADi
        Q = 1. * np.eye(n_states)
        R = 0.1 * np.eye(n_inputs)
        # Solve the Algebraic Riccati Equation
        P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

        swi_lin.load_weights(Q, R, P)
        swi_lin_casadi.precompute_matrices(x0, Q, R, P)
        # print("Precomputation complete!")
        
        # print("Computing cost function value using Casadi SwiLin...")
        x0_aug = np.concatenate([x0, [1]])
        cost_casadi = swi_lin_casadi.cost_function(R, x0_aug)
        # Flatten controls and durations into individual scalar arguments expected by CasADi
        controls = self.controls_gt.to(self.device)
        phases_duration = self.phases_duration_gt.to(self.device)
        controls_args = [float(controls[0, i]) for i in range(n_phases)]
        duration_args = [float(phases_duration[i]) for i in range(n_phases)]
        cost_casadi_value = cost_casadi(*controls_args, *duration_args).full().item()
        print(f"Cost value (Casadi SwiLin): {cost_casadi_value:.6f}")
        
        print("Computing the cost function using the model output...")
        with torch.no_grad():
            x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0)

            prediction = self.model(x0)
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(self.model.sys.time_horizon, device=prediction.device, dtype=prediction.dtype)

            # Handle batch dimension properly
            n_control_outputs = self.model.n_phases * n_inputs
            pred_u = prediction[:, :n_control_outputs] # shape (batch_size, n_phases * n_inputs)
            pred_delta_raw = prediction[:, n_control_outputs:]
            
            # Apply softmax and scale deltas
            delta_normalized = F.softmax(pred_delta_raw, dim=-1)
            pred_deltas = delta_normalized * T_tensor # shape (batch_size, n_phases)
            
            # Clip controls using tanh-based soft clipping to preserve gradients
            u_min = -1.0  # Define your lower bound
            u_max = 1.0   # Define your upper bound
            u_center = (u_max + u_min) / 2.0
            u_range = (u_max - u_min) / 2.0
            # Soft clipping: maps (-inf, inf) to (u_min, u_max) smoothly
            pred_u = u_center + u_range * torch.tanh(pred_u)
            
            # Reshape for cost_function: remove batch dimension and reshape controls
            # pred_u: (batch_size, n_phases * n_inputs) -> (n_phases, n_inputs)
            # pred_deltas: (batch_size, n_phases) -> (n_phases,)
            pred_u_reshaped = pred_u.squeeze(0).reshape(n_phases, n_inputs)
            pred_deltas_squeezed = pred_deltas.squeeze(0)
            x0_squeezed = x0.squeeze(0)
            
            cost_data = swi_lin.cost_function(pred_u_reshaped, pred_deltas_squeezed, x0_squeezed)
            
        return cost_casadi_value, cost_data.item()
    
    def plot_network_output(self, x0, save_path: Optional[str] = None, show_ground_truth: bool = True) -> None:
        """
        Plot the neural network output for controls and phase durations.
        
        Args:
            x0: Initial state tensor.
            save_path (str, optional): Path to save the figure. If None, the plot is displayed.
            show_ground_truth (bool): Whether to show ground truth data if available.
        """
        n_inputs = self.n_inputs
        n_phases = self.n_phases
        
        with torch.no_grad():
            x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0)
            
            prediction = self.model(x0)
            
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(self.model.sys.time_horizon, device=prediction.device, dtype=prediction.dtype)
            
            # Handle batch dimension properly
            n_control_outputs = self.model.n_phases * n_inputs
            pred_u = prediction[:, :n_control_outputs]
            pred_delta_raw = prediction[:, n_control_outputs:]
            
            # Apply softmax and scale deltas
            delta_normalized = F.softmax(pred_delta_raw, dim=-1)
            pred_deltas = delta_normalized * T_tensor
            
            # Clip controls using tanh-based soft clipping
            u_min = -1.0
            u_max = 1.0
            u_center = (u_max + u_min) / 2.0
            u_range = (u_max - u_min) / 2.0
            pred_u = u_center + u_range * torch.tanh(pred_u)
            
            # Reshape predictions for plotting
            pred_u_reshaped = pred_u.squeeze(0).reshape(n_phases, n_inputs).cpu().numpy()
            pred_deltas_np = pred_deltas.squeeze(0).cpu().numpy()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot controls
        for i in range(n_inputs):
            ax1.plot(range(n_phases), pred_u_reshaped[:, i], 'o-', label=rf'Predicted $u_{{{i+1}}}$', linewidth=2)
            if show_ground_truth and hasattr(self, 'controls_gt'):
                controls_gt_np = self.controls_gt.cpu().numpy()
                ax1.plot(range(n_phases), controls_gt_np[i, :], 's--', label=rf'Ground Truth $u_{{{i+1}}}$', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Phase Index $k$', fontsize=12)
        ax1.set_ylabel(r'Control Value $u_k$', fontsize=12)
        ax1.set_title('Control Inputs per Phase', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot phase durations
        ax2.bar(range(n_phases), pred_deltas_np, alpha=0.7, label=r'Predicted $\delta_k$', color='steelblue')
        if show_ground_truth and hasattr(self, 'phases_duration_gt'):
            phases_duration_gt_np = self.phases_duration_gt.cpu().numpy()
            ax2.bar(range(n_phases), phases_duration_gt_np, alpha=0.5, label=r'Ground Truth $\delta_k$', color='orange')
        
        ax2.set_xlabel('Phase Index $k$', fontsize=12)
        ax2.set_ylabel(r'Duration $\delta_k$', fontsize=12)
        ax2.set_title('Phase Durations', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_piecewise_constant_control(self, x0, save_path: Optional[str] = None, show_ground_truth: bool = True) -> None:
        """
        Plot the controls as piecewise constant signals over time.
        Each control value is held constant for its corresponding phase duration.
        
        Args:
            x0: Initial state tensor.
            save_path (str, optional): Path to save the figure. If None, the plot is displayed.
            show_ground_truth (bool): Whether to show ground truth data if available.
        """
        n_inputs = self.n_inputs
        n_phases = self.n_phases
        
        with torch.no_grad():
            x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0)
            
            prediction = self.model(x0)
            
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(self.model.sys.time_horizon, device=prediction.device, dtype=prediction.dtype)
            
            # Handle batch dimension properly
            n_control_outputs = self.model.n_phases * n_inputs
            pred_u = prediction[:, :n_control_outputs]
            pred_delta_raw = prediction[:, n_control_outputs:]
            
            # Apply softmax and scale deltas
            delta_normalized = F.softmax(pred_delta_raw, dim=-1)
            pred_deltas = delta_normalized * T_tensor
            
            # Clip controls using tanh-based soft clipping
            u_min = -1.0
            u_max = 1.0
            u_center = (u_max + u_min) / 2.0
            u_range = (u_max - u_min) / 2.0
            pred_u = u_center + u_range * torch.tanh(pred_u)
            
            # Reshape predictions for plotting
            pred_u_reshaped = pred_u.squeeze(0).reshape(n_phases, n_inputs).cpu().numpy()
            pred_deltas_np = pred_deltas.squeeze(0).cpu().numpy()
        
        # Create time points for piecewise constant signal
        time_points_pred = np.concatenate([[0], np.cumsum(pred_deltas_np)])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot predicted controls as piecewise constant
        for i in range(n_inputs):
            for phase in range(n_phases):
                t_start = time_points_pred[phase]
                t_end = time_points_pred[phase + 1]
                control_value = pred_u_reshaped[phase, i]
                
                # Draw horizontal line for this phase
                ax.plot([t_start, t_end], [control_value, control_value], 
                       linewidth=2.5, color=f'C0', label=rf'Predicted $u_{{{i+1}}}(t)$' if phase == 0 else '')
                
                # Draw vertical line at phase transition (except at start)
                if phase > 0:
                    prev_value = pred_u_reshaped[phase - 1, i]
                    ax.plot([t_start, t_start], [prev_value, control_value], 
                           linewidth=2.5, color=f'C0', linestyle=':')
        
        # Plot ground truth if available
        if show_ground_truth and hasattr(self, 'controls_gt') and hasattr(self, 'phases_duration_gt'):
            controls_gt_np = self.controls_gt.cpu().numpy()
            phases_duration_gt_np = self.phases_duration_gt.cpu().numpy()
            time_points_gt = np.concatenate([[0], np.cumsum(phases_duration_gt_np)])
            
            for i in range(n_inputs):
                for phase in range(n_phases):
                    t_start = time_points_gt[phase]
                    t_end = time_points_gt[phase + 1]
                    control_value = controls_gt_np[i, phase]
                    
                    # Draw horizontal line for this phase
                    ax.plot([t_start, t_end], [control_value, control_value], 
                           linewidth=2, color=f'C1', linestyle='--', alpha=0.6,
                           label=rf'Ground Truth $u_{{{i+1}}}(t)$' if phase == 0 else '')
                    
                    # Draw vertical line at phase transition (except at start)
                    if phase > 0:
                        prev_value = controls_gt_np[i, phase - 1]
                        ax.plot([t_start, t_start], [prev_value, control_value], 
                               linewidth=2, color=f'C1', linestyle=':', alpha=0.6)
        
        ax.set_xlabel(r'Time $t$', fontsize=12)
        ax.set_ylabel(r'Control $u(t)$', fontsize=12)
        ax.set_title('Piecewise Constant Control Signal Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    
if __name__ == "__main__":
    ## Example usage of ModelValidator
    
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_phases = 80
    n_control_inputs = 1
    n_NN_inputs = 3
    n_NN_outputs = n_phases * (n_control_inputs + 1)
    model = SwiLinNN(
        layer_sizes=[n_NN_inputs, 128, 256, n_NN_outputs],
        n_phases=n_phases,
    )
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'pannocchia_torch_20251209_112339.pt'))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model to evaluation mode
    model.eval()
    
    # Create the model validator
    validator = ModelValidator(model, device=device)
    
    # Load data
    validator.load_data('optimal_params.mat')
    
    # Define initial state
    x0 = np.array([1.3440, -4.5850, 5.6470])
    
    # Validate on data
    criterion = nn.MSELoss()
    loss = validator.validate_on_data(x0, criterion)
    print(f"Validation loss on data: {loss:.6f}")
    
    # Validate on custom function
    cost_casadi, cost_data = validator.validate_on_custom_function(x0)
    print(f"Cost comparison - Casadi: {cost_casadi:.6f}, Data: {cost_data:.6f}")
    
    # Plot the network output
    print("\nGenerating plot...")
    # plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'network_output.png'))
    # validator.plot_network_output(x0, save_path=None, show_ground_truth=True)
    
    # Plot piecewise constant control
    print("\nGenerating piecewise constant control plot...")
    # piecewise_plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'piecewise_control.png'))
    validator.plot_piecewise_constant_control(x0, save_path=None, show_ground_truth=True)