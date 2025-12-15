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
from .switched_linear_torch import SwiLin
from .training import SwiLinNN


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
            float: Computed loss value.
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

    def validate_on_cost_function(self, x0) -> Dict[str, float]:
        """
        Validate the cost function value based on the output of the model.

        """
        # First compute the cost function value using the Casadi SwiLin system
        # Create a simple switched linear system
        n_phases = 50
        n_states = 3
        n_inputs = 2
        time_horizon = 2.0

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

        # print("Creating SwiLin system...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        swi_lin = SwiLin(n_phases, n_states, n_inputs, time_horizon, auto=False, device=device) 
        swi_lin_casadi = SwiLin_casadi(n_phases, n_states, n_inputs, time_horizon, auto=False)
        swi_lin.load_model(model)
        swi_lin_casadi.load_model(model)

        # print("Precomputing matrices...")
        # Convert x0 to numpy for CasADi
        Q = 10. * np.eye(n_states)
        R = 10. * np.eye(n_inputs)
        E = 1. * np.eye(n_states)

        swi_lin.load_weights(Q, R, E)
        swi_lin_casadi.precompute_matrices(x0, Q, R, E)
        # print("Precomputation complete!")
        
        # print("Computing cost function value using Casadi SwiLin...")
        x0_aug = np.concatenate([x0, [1]])
        cost_casadi = swi_lin_casadi.cost_function(R, x0_aug)
        # Flatten controls and durations into individual scalar arguments expected by CasADi
        controls = self.controls_gt.to(self.device)
        phases_duration = self.phases_duration_gt.to(self.device)
        # Flatten controls into phase-major order (for each phase, list all inputs)
        controls_args = []
        for i in range(n_phases):
            for k in range(n_inputs):
                controls_args.append(float(controls[k, i]))
        duration_args = [float(phases_duration[i]) for i in range(n_phases)]
        # The CasADi Function may expect a different flattening/order of control inputs
        # depending on the implementation/version. Try the multi-input flattening first;
        # if CasADi raises an "Incorrect number of inputs" error, fall back to the
        # original phase-major single-input ordering (first input only) to preserve
        # backward compatibility with older casadi wrappers.
        try:
            cost_casadi_value = cost_casadi(*controls_args, *duration_args).full().item()
        except RuntimeError as e:
            msg = str(e)
            if 'Incorrect number of inputs' in msg or 'arg.size()==n_in_' in msg:
                # Fallback: use only the first input per phase (legacy behavior)
                fallback_controls = [float(controls[0, i]) for i in range(n_phases)]
                try:
                    cost_casadi_value = cost_casadi(*fallback_controls, *duration_args).full().item()
                except Exception:
                    # Re-raise the original exception to avoid hiding unexpected errors
                    raise
            else:
                raise
        # print(f"Cost value (Casadi SwiLin): {cost_casadi_value:.6f}")
        
        # print("Computing the cost function using the model output...")
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
    
    def check_constraints(self, x0, u_min: float = -1.0, u_max: float = 1.0, verbose: bool = True) -> Dict[str, any]:
        """
        Check constraint violations for controls and phase durations.
        
        Args:
            x0: Initial state tensor.
            u_min (float): Minimum control value bound.
            u_max (float): Maximum control value bound.
            verbose (bool): Whether to print detailed constraint violation information.
            
        Returns:
            dict: Dictionary containing constraint violation information:
                - 'controls_in_bounds': bool, whether all controls satisfy bounds
                - 'control_violations': dict with min/max violations
                - 'durations_in_bounds': bool, whether all durations are in [0, T]
                - 'duration_violations': dict with violations
                - 'sum_constraint': bool, whether sum of durations equals T
                - 'duration_sum': float, actual sum of durations
                - 'duration_sum_error': float, error in sum constraint
                - 'all_constraints_satisfied': bool, overall constraint satisfaction
        """
        n_inputs = self.n_inputs
        n_phases = self.n_phases
        T = self.model.sys.time_horizon
        
        with torch.no_grad():
            x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0)
            
            prediction = self.model(x0)
            
            # Apply transformation: T * softmax(output[-n_phases:]) for the deltas
            T_tensor = torch.tensor(T, device=prediction.device, dtype=prediction.dtype)
            
            # Handle batch dimension properly
            n_control_outputs = self.model.n_phases * n_inputs
            pred_u = prediction[:, :n_control_outputs]
            pred_delta_raw = prediction[:, n_control_outputs:]
            
            # Apply softmax and scale deltas
            delta_normalized = F.softmax(pred_delta_raw, dim=-1)
            pred_deltas = delta_normalized * T_tensor
            
            # Clip controls using tanh-based soft clipping
            u_center = (u_max + u_min) / 2.0
            u_range = (u_max - u_min) / 2.0
            pred_u = u_center + u_range * torch.tanh(pred_u)
            
            # Reshape predictions
            pred_u_reshaped = pred_u.squeeze(0).reshape(n_phases, n_inputs).cpu().numpy()
            pred_deltas_np = pred_deltas.squeeze(0).cpu().numpy()
        
        # Check control constraints
        u_min_violation = np.minimum(pred_u_reshaped - u_min, 0.0)
        u_max_violation = np.maximum(pred_u_reshaped - u_max, 0.0)
        
        controls_in_bounds = np.all(u_min_violation == 0.0) and np.all(u_max_violation == 0.0)
        
        # Check duration constraints
        duration_lower_violation = np.minimum(pred_deltas_np, 0.0)
        duration_upper_violation = np.maximum(pred_deltas_np - T, 0.0)
        
        durations_in_bounds = np.all(duration_lower_violation == 0.0) and np.all(duration_upper_violation == 0.0)
        
        # Check sum constraint
        duration_sum = np.sum(pred_deltas_np)
        duration_sum_error = duration_sum - T
        sum_constraint_satisfied = np.abs(duration_sum_error) < 1e-6
        
        all_constraints_satisfied = controls_in_bounds and durations_in_bounds and sum_constraint_satisfied
        
        # Per-input statistics (useful when n_inputs > 1)
        per_input_stats = []
        if n_inputs > 0:
            per_input_min = pred_u_reshaped.min(axis=0)
            per_input_max = pred_u_reshaped.max(axis=0)
            per_input_min_violation = np.minimum(pred_u_reshaped - u_min, 0.0).min(axis=0)
            per_input_max_violation = np.maximum(pred_u_reshaped - u_max, 0.0).max(axis=0)
            for k in range(n_inputs):
                per_input_stats.append({
                    'input_index': k,
                    'min_value': float(per_input_min[k]),
                    'max_value': float(per_input_max[k]),
                    'min_violation': float(per_input_min_violation[k]),
                    'max_violation': float(per_input_max_violation[k]),
                })

        results = {
            'controls_in_bounds': controls_in_bounds,
            'control_violations': {
                'min_violation': float(np.min(u_min_violation)),
                'max_violation': float(np.max(u_max_violation)),
                'min_value': float(np.min(pred_u_reshaped)),
                'max_value': float(np.max(pred_u_reshaped)),
                'per_input': per_input_stats,
            },
            'durations_in_bounds': durations_in_bounds,
            'duration_violations': {
                'lower_violation': float(np.min(duration_lower_violation)),
                'upper_violation': float(np.max(duration_upper_violation)),
                'min_value': float(np.min(pred_deltas_np)),
                'max_value': float(np.max(pred_deltas_np)),
            },
            'sum_constraint': sum_constraint_satisfied,
            'duration_sum': float(duration_sum),
            'duration_sum_error': float(duration_sum_error),
            'time_horizon': float(T),
            'all_constraints_satisfied': all_constraints_satisfied,
        }
        
        if verbose:
            print("="*60)
            print("CONSTRAINT VIOLATION CHECK")
            print("="*60)
            print(f"\nControl Constraints (bounds: [{u_min}, {u_max}]):")
            print(f"  ✓ In bounds: {controls_in_bounds}")
            # If multiple inputs, print per-input statistics
            if n_inputs > 1:
                print(f"  Per-input control stats:")
                for info in results['control_violations']['per_input']:
                    idx = info['input_index'] + 1
                    print(f"    - Input {idx}: min={info['min_value']:.6f}, max={info['max_value']:.6f}")
                    if not controls_in_bounds:
                        if info['min_violation'] < 0:
                            print(f"      ⚠ Min violation: {info['min_violation']:.6f}")
                        if info['max_violation'] > 0:
                            print(f"      ⚠ Max violation: {info['max_violation']:.6f}")
            else:
                print(f"  Min control value: {results['control_violations']['min_value']:.6f}")
                print(f"  Max control value: {results['control_violations']['max_value']:.6f}")
                if not controls_in_bounds:
                    print(f"  ⚠ Min violation: {results['control_violations']['min_violation']:.6f}")
                    print(f"  ⚠ Max violation: {results['control_violations']['max_violation']:.6f}")
            
            print(f"\nPhase Duration Constraints (bounds: [0, {T}]):")
            print(f"  ✓ In bounds: {durations_in_bounds}")
            print(f"  Min duration: {results['duration_violations']['min_value']:.6f}")
            print(f"  Max duration: {results['duration_violations']['max_value']:.6f}")
            if not durations_in_bounds:
                print(f"  ⚠ Lower bound violation: {results['duration_violations']['lower_violation']:.6f}")
                print(f"  ⚠ Upper bound violation: {results['duration_violations']['upper_violation']:.6f}")
            
            print(f"\nDuration Sum Constraint (should equal {T}):")
            print(f"  ✓ Satisfied: {sum_constraint_satisfied}")
            print(f"  Sum of durations: {results['duration_sum']:.6f}")
            print(f"  Error: {results['duration_sum_error']:.6e}")
            
            print(f"\n{'='*60}")
            if all_constraints_satisfied:
                print("✓ ALL CONSTRAINTS SATISFIED")
            else:
                print("⚠ SOME CONSTRAINTS VIOLATED")
            print("="*60)
        
        return results
    
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
        
        # If multiple inputs, create one subplot per input for controls, plus one for durations
        if n_inputs <= 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            # Single-input plotting (keeps previous style)
            ax1.plot(range(n_phases), pred_u_reshaped[:, 0], 'o-', label=rf'Predicted $u_{{1}}$', linewidth=2)
            if show_ground_truth and hasattr(self, 'controls_gt'):
                controls_gt_np = self.controls_gt.cpu().numpy()
                ax1.plot(range(n_phases), controls_gt_np[0, :], 's--', label=rf'Ground Truth $u_{{1}}$', linewidth=2, alpha=0.7)
            ax1.set_xlabel('Phase Index $k$', fontsize=12)
            ax1.set_ylabel(r'Control Value $u_k$', fontsize=12)
            ax1.set_title('Control Inputs per Phase', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # durations
            ax2.bar(range(n_phases), pred_deltas_np, alpha=0.7, label=r'Predicted $\delta_k$', color='steelblue')
            if show_ground_truth and hasattr(self, 'phases_duration_gt'):
                phases_duration_gt_np = self.phases_duration_gt.cpu().numpy()
                ax2.bar(range(n_phases), phases_duration_gt_np, alpha=0.5, label=r'Ground Truth $\delta_k$', color='orange')
            ax2.set_xlabel('Phase Index $k$', fontsize=12)
            ax2.set_ylabel(r'Duration $\delta_k$', fontsize=12)
            ax2.set_title('Phase Durations', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            # Create one axis per input + one for durations
            fig, axes = plt.subplots(n_inputs + 1, 1, figsize=(12, 3 * (n_inputs + 1)), sharex=False)
            # axes[0..n_inputs-1] are for each input
            controls_gt_np = self.controls_gt.cpu().numpy() if (show_ground_truth and hasattr(self, 'controls_gt')) else None
            for i in range(n_inputs):
                ax = axes[i]
                ax.plot(range(n_phases), pred_u_reshaped[:, i], 'o-', label=rf'Predicted $u_{{{i+1}}}$', linewidth=2)
                if controls_gt_np is not None:
                    ax.plot(range(n_phases), controls_gt_np[i, :], 's--', label=rf'Ground Truth $u_{{{i+1}}}$', linewidth=2, alpha=0.7)
                ax.set_ylabel(rf'$u_{{{i+1}}}$', fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            # durations in the last axis
            axd = axes[-1]
            axd.bar(range(n_phases), pred_deltas_np, alpha=0.7, label=r'Predicted $\delta_k$', color='steelblue')
            if show_ground_truth and hasattr(self, 'phases_duration_gt'):
                phases_duration_gt_np = self.phases_duration_gt.cpu().numpy()
                axd.bar(range(n_phases), phases_duration_gt_np, alpha=0.5, label=r'Ground Truth $\delta_k$', color='orange')
            axd.set_xlabel('Phase Index $k$', fontsize=12)
            axd.set_ylabel(r'$\delta_k$', fontsize=11)
            axd.set_title('Phase Durations', fontsize=14, fontweight='bold')
            axd.legend(fontsize=9)
            axd.grid(True, alpha=0.3, axis='y')
        
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
        
        # Create figure(s): one subplot per input for piecewise control
        if n_inputs <= 1:
            fig, ax = plt.subplots(figsize=(14, 6))
            for phase in range(n_phases):
                t_start = time_points_pred[phase]
                t_end = time_points_pred[phase + 1]
                control_value = pred_u_reshaped[phase, 0]
                ax.plot([t_start, t_end], [control_value, control_value], linewidth=2.5, color='C0')
                if phase > 0:
                    prev_value = pred_u_reshaped[phase - 1, 0]
                    ax.plot([t_start, t_start], [prev_value, control_value], linewidth=2.5, color='C0', linestyle=':')

            # Ground truth
            if show_ground_truth and hasattr(self, 'controls_gt') and hasattr(self, 'phases_duration_gt'):
                controls_gt_np = self.controls_gt.cpu().numpy()
                phases_duration_gt_np = self.phases_duration_gt.cpu().numpy()
                time_points_gt = np.concatenate([[0], np.cumsum(phases_duration_gt_np)])
                for phase in range(n_phases):
                    t_start = time_points_gt[phase]
                    t_end = time_points_gt[phase + 1]
                    control_value = controls_gt_np[0, phase]
                    ax.plot([t_start, t_end], [control_value, control_value], linewidth=2, color='C1', linestyle='--', alpha=0.6)
                    if phase > 0:
                        prev_value = controls_gt_np[0, phase - 1]
                        ax.plot([t_start, t_start], [prev_value, control_value], linewidth=2, color='C1', linestyle=':', alpha=0.6)

            ax.set_xlabel(r'Time $t$', fontsize=12)
            ax.set_ylabel(r'Control $u(t)$', fontsize=12)
            ax.set_title('Piecewise Constant Control Signal Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend([r'Predicted $u(t)$', r'Ground Truth $u(t)$'] if (show_ground_truth and hasattr(self, 'controls_gt')) else [r'Predicted $u(t)$'], fontsize=10, loc='best')
        else:
            fig, axes = plt.subplots(n_inputs, 1, figsize=(14, 3 * n_inputs), sharex=False)
            controls_gt_np = self.controls_gt.cpu().numpy() if (show_ground_truth and hasattr(self, 'controls_gt')) else None
            phases_duration_gt_np = self.phases_duration_gt.cpu().numpy() if (show_ground_truth and hasattr(self, 'phases_duration_gt')) else None
            time_points_gt = np.concatenate([[0], np.cumsum(phases_duration_gt_np)]) if phases_duration_gt_np is not None else None

            for i in range(n_inputs):
                ax = axes[i]
                for phase in range(n_phases):
                    t_start = time_points_pred[phase]
                    t_end = time_points_pred[phase + 1]
                    control_value = pred_u_reshaped[phase, i]
                    ax.plot([t_start, t_end], [control_value, control_value], linewidth=2.5, color='C0')
                    if phase > 0:
                        prev_value = pred_u_reshaped[phase - 1, i]
                        ax.plot([t_start, t_start], [prev_value, control_value], linewidth=2.5, color='C0', linestyle=':')

                if controls_gt_np is not None and time_points_gt is not None:
                    for phase in range(n_phases):
                        t_start = time_points_gt[phase]
                        t_end = time_points_gt[phase + 1]
                        control_value = controls_gt_np[i, phase]
                        ax.plot([t_start, t_end], [control_value, control_value], linewidth=2, color='C1', linestyle='--', alpha=0.6)
                        if phase > 0:
                            prev_value = controls_gt_np[i, phase - 1]
                            ax.plot([t_start, t_start], [prev_value, control_value], linewidth=2, color='C1', linestyle=':', alpha=0.6)

                ax.set_ylabel(rf'$u_{{{i+1}}}(t)$', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.legend([rf'Predicted $u_{{{i+1}}}(t)$', rf'Ground Truth $u_{{{i+1}}}(t)$'] if controls_gt_np is not None else [rf'Predicted $u_{{{i+1}}}(t)$'], fontsize=9)
        
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
    
    n_phases = 50
    n_control_inputs = 2
    n_NN_inputs = 3
    n_NN_outputs = n_phases * (n_control_inputs + 1)
    model = SwiLinNN(
        layer_sizes=[n_NN_inputs, 50, 50, n_NN_outputs],
        n_phases=n_phases,
    )
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'nahs_torch_20251212_165742.pt'))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model to evaluation mode
    model.eval()
    
    # Create the model validator
    validator = ModelValidator(model, device=device)
    
    # Load data
    validator.load_data('example_1_paper_NAHS.mat')
    
    # Define initial state
    x0 = np.array([2, -1, 5])
    
    # Validate on data
    criterion = nn.MSELoss()
    loss = validator.validate_on_data(x0, criterion)
    print("="*60)
    print("COMPARING MODEL OUTPUT TO GROUND TRUTH DATA")
    print("="*60)
    print(f"\nMSE data w.r.t. ground truth: {loss:.6f}\n")
    
    # Validate on custom function
    cost_casadi, cost_data = validator.validate_on_cost_function(x0)
    print("="*60)
    print("COMPARING COST FUNCTION VALUES")
    print("="*60)
    print(f"\nCost comparison - Casadi: {cost_casadi:.6f}, Neural Network: {cost_data:.6f}")
    print(f"Cost difference: {abs(cost_casadi - cost_data):.6f}")
    print(f"Relative error: {abs(cost_casadi - cost_data) / abs(cost_casadi):.6f}\n")
    
    # Check constraints
    constraint_results = validator.check_constraints(x0, u_min=-1.0, u_max=1.0, verbose=True)
    
    # Plot the network output
    print("\nGenerating plot...")
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images', 'network_output.png'))
    validator.plot_network_output(x0, save_path=None, show_ground_truth=True)
    
    # Plot piecewise constant control
    print("\nGenerating piecewise constant control plot...")
    piecewise_plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images', 'piecewise_control.png'))
    validator.plot_piecewise_constant_control(x0, save_path=None, show_ground_truth=True)