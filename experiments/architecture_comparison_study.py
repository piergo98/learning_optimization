"""
Architecture Comparison Study

This script trains multiple neural networks (SwiLinNN) with varying architectures:
- Different numbers of layers
- Different numbers of neurons per layer
- Different total parameter counts

The goal is to build statistics evaluating how architecture choices affect:
1. Cost function value
2. Control inputs (u)
3. Mode durations (delta)

All networks are trained on the same initial states and compared against the optimization solution.
"""

import json
import warnings
import os
import sys
import time
from typing import Optional, Callable, Tuple, Dict, List
from datetime import datetime
import itertools
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy.linalg import solve_continuous_are
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU training will not be available.")
    
from ocslc.switched_linear_mpc import SwitchedLinearMPC as SwiLin_casadi

from src.switched_linear_torch import SwiLin
from src.training import SwiLinNN, train_neural_network, evaluate_cost_functional


# ============================================================================
# Configuration
# ============================================================================

class ExperimentConfig:
    """Configuration for the architecture comparison study."""
    
    # Problem setup
    N_PHASES = 10
    TIME_HORIZON = 1.0
    N_CONTROL_INPUTS = 1
    N_STATES = 1
    N_NN_INPUTS = 1
    
    # Casadi settings
    MULTIPLE_SHOOTING = True
    INTEGRATOR = 'exp'
    HYBRID = False
    PLOT = 'display'
    
    # Training setup
    DEVICE = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    N_SAMPLES_TRAIN = 1000
    N_SAMPLES_VAL = 100
    N_SAMPLES_TEST = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 200
    BATCH_SIZE = N_SAMPLES_TRAIN
    OPTIMIZER = 'adam'
    
    # Data range
    X_MIN = -5.0
    X_MAX = 5.0
    
    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 30
    EARLY_STOPPING_MIN_DELTA = 1e-6
    
    # Save paths
    EXPERIMENT_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'architecture_study')
    RESULTS_DIR = os.path.join(EXPERIMENT_DIR, 'results')
    MODELS_DIR = os.path.join(EXPERIMENT_DIR, 'models')
    PLOTS_DIR = os.path.join(EXPERIMENT_DIR, 'plots')
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the experiment."""
        for directory in [cls.EXPERIMENT_DIR, cls.RESULTS_DIR, cls.MODELS_DIR, cls.PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)


# ============================================================================
# Architecture Definitions
# ============================================================================

def generate_architectures(n_inputs: int, n_outputs: int) -> List[Dict]:
    """
    Generate a comprehensive list of architectures to test.
    
    Returns a list of dictionaries, each containing:
    - 'name': descriptive name
    - 'layer_sizes': list of layer sizes
    - 'n_layers': number of hidden layers
    - 'neurons_per_layer': average neurons per hidden layer
    - 'n_params': total number of parameters (computed later)
    """
    architectures = []
    
    # 1. Varying depth (keeping width constant)
    base_width = 64
    for n_hidden in [1, 2, 3, 4, 5]:
        hidden_layers = [base_width] * n_hidden
        layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
        architectures.append({
            'name': f'depth_{n_hidden}_width_{base_width}',
            'layer_sizes': layer_sizes,
            'n_hidden_layers': n_hidden,
            'width': base_width,
        })
    
    # 2. Varying width (keeping depth constant)
    n_hidden = 3
    for width in [16, 32, 64, 128, 256]:
        hidden_layers = [width] * n_hidden
        layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
        architectures.append({
            'name': f'depth_{n_hidden}_width_{width}',
            'layer_sizes': layer_sizes,
            'n_hidden_layers': n_hidden,
            'width': width,
        })
    
    # 3. Pyramidal architectures (decreasing width)
    for start_width in [128, 256]:
        for n_hidden in [3, 4]:
            hidden_layers = [start_width // (2**i) for i in range(n_hidden)]
            layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
            architectures.append({
                'name': f'pyramidal_{n_hidden}_start_{start_width}',
                'layer_sizes': layer_sizes,
                'n_hidden_layers': n_hidden,
                'width': 'pyramidal',
            })
    
    # 4. Inverted pyramidal (increasing width)
    for end_width in [128, 256]:
        for n_hidden in [3, 4]:
            hidden_layers = [end_width // (2**(n_hidden-1-i)) for i in range(n_hidden)]
            layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
            architectures.append({
                'name': f'inv_pyramidal_{n_hidden}_end_{end_width}',
                'layer_sizes': layer_sizes,
                'n_hidden_layers': n_hidden,
                'width': 'inv_pyramidal',
            })
    
    # 5. Small networks
    for layer_sizes in [[n_inputs, 8, n_outputs], 
                        [n_inputs, 16, 16, n_outputs],
                        [n_inputs, 32, 16, 8, n_outputs]]:
        n_hidden = len(layer_sizes) - 2
        avg_width = int(np.mean(layer_sizes[1:-1]))
        architectures.append({
            'name': f'small_{n_hidden}_{avg_width}',
            'layer_sizes': layer_sizes,
            'n_hidden_layers': n_hidden,
            'width': avg_width,
        })
    
    # 6. Large networks
    for layer_sizes in [[n_inputs, 512, n_outputs],
                        [n_inputs, 256, 256, n_outputs],
                        [n_inputs, 256, 256, 256, n_outputs]]:
        n_hidden = len(layer_sizes) - 2
        avg_width = int(np.mean(layer_sizes[1:-1]))
        architectures.append({
            'name': f'large_{n_hidden}_{avg_width}',
            'layer_sizes': layer_sizes,
            'n_hidden_layers': n_hidden,
            'width': avg_width,
        })
    
    return architectures


def count_parameters(network: nn.Module) -> int:
    """Count total number of trainable parameters in a network."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


# ============================================================================
# Optimization Baseline (Ground Truth)
# ============================================================================

def compute_optimal_solution(x0: np.ndarray, config: ExperimentConfig) -> Dict:
    """
    Compute the optimal solution using gradient-based optimization.
    
    This serves as ground truth for comparison.
    """
    # Create SwiLin system
    model = {
        'A': [np.array([[1]]), np.array([[-3]])],
        'B': [np.array([[2]]), np.array([[-1]])],
    }
    
    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]
    
    swi_lin_casadi = SwiLin_casadi(
        model, 
        n_phases=config.N_PHASES, 
        time_horizon=config.TIME_HORIZON, 
        auto=False,
        x0=x0,
        multiple_shooting=config.MULTIPLE_SHOOTING,
        propagation=config.INTEGRATOR,
        inspect = False,
        hybrid=config.HYBRID,
        plot=config.PLOT,
    )
    
    swi_lin_casadi.load_model(model)
    
    Q = 1.0 * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    E = 0 * np.eye(n_states)

    swi_lin_casadi.precompute_matrices(x0, Q, R, E)

    # loaded_data = scipy.io.loadmat('optimal_results_hybrid.mat')
    # fixed_states = loaded_data['trajectory'][0]
    # fixed_inputs = loaded_data['controls'][0]

    states_lb = np.array([config.X_MIN])
    states_ub = np.array([config.X_MAX]) 

    swi_lin_casadi.set_bounds(
        -1, 
        1, 
        states_lb, 
        states_ub, 
        # inspect_inputs=fixed_inputs,
        # inspect_states=fixed_states,
    )

    if swi_lin_casadi.multiple_shooting:
        swi_lin_casadi.multiple_shooting_constraints(x0)

    swi_lin_casadi.set_cost_function(Q, R, x0, E)

    # Set the initial guess  
    exp_dist = 0.99**np.arange(80)
    phase_durations = exp_dist * config.TIME_HORIZON / np.sum(exp_dist)

    swi_lin_casadi.set_initial_guess(
        x0, 
        # initial_phases_duration=phase_durations
    )

    swi_lin_casadi.create_solver(
        'ipopt',
        print_level=0,
        print_time=False,
    )

    u_opt, delta_opt, states_opt = swi_lin_casadi.solve()
    cost_opt = swi_lin_casadi.opt_cost
    
    return {
        'u': np.array(u_opt),
        'delta': np.array(delta_opt),
        'cost': cost_opt.item(),
        'x0': x0
    }


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_architecture(
    arch: Dict,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    config: ExperimentConfig,
    verbose: bool = True
) -> Tuple[SwiLinNN, Dict]:
    """
    Train a single architecture and return the trained network and history.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training: {arch['name']}")
        print(f"Architecture: {arch['layer_sizes']}")
    
    # Create network
    network = SwiLinNN(
        layer_sizes=arch['layer_sizes'],
        n_phases=config.N_PHASES,
        activation='relu',
        output_activation='linear'
    )
    
    # Count parameters
    n_params = count_parameters(network)
    arch['n_params'] = n_params
    
    if verbose:
        print(f"Total parameters: {n_params:,}")
    
    # Training directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_dir = os.path.join(config.RESULTS_DIR, f"{arch['name']}_{timestamp}")
    os.makedirs(train_dir, exist_ok=True)
    
    # Train
    start_time = time.time()
    params_opt, history = train_neural_network(
        network=network,
        X_train=X_train,
        X_val=X_val,
        optimizer=config.OPTIMIZER,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        n_epochs=config.N_EPOCHS,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        verbose=False,
        tensorboard_logdir=train_dir,
        save_history=True,
        save_history_path=os.path.join(train_dir, 'history.json'),
        save_model=True,
        save_model_path=os.path.join(config.MODELS_DIR, f"{arch['name']}_{timestamp}.pt"),
        early_stopping=config.EARLY_STOPPING,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=config.EARLY_STOPPING_MIN_DELTA,
    )
    training_time = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {training_time:.2f}s")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Add metadata to history
    history['training_time'] = training_time
    history['n_params'] = n_params
    history['architecture'] = arch
    
    return network, history


def evaluate_network_on_test_set(
    network: SwiLinNN,
    X_test: torch.Tensor,
    optimal_solutions: List[Dict],
    config: ExperimentConfig
) -> Dict:
    """
    Evaluate a trained network on the test set and compare with optimal solutions.
    """
    network.eval()
    device = config.DEVICE
    
    results = {
        'costs_nn': [],
        'costs_opt': [],
        'u_errors': [],
        'delta_errors': [],
        'cost_diff': [],
        'u_mse': [],
        'delta_mse': [],
    }
    
    with torch.no_grad():
        for i, x0_tensor in enumerate(X_test):
            # Get optimal solution
            opt_sol = optimal_solutions[i]
            
            # Get network prediction
            x0_input = x0_tensor.to(device).unsqueeze(0)
            output = network(x0_input).squeeze(0)
            
            # Extract and transform outputs
            n_control_outputs = config.N_PHASES * config.N_CONTROL_INPUTS
            pred_u_raw = output[:n_control_outputs]
            pred_delta_raw = output[n_control_outputs:]
            
            # Apply transformations (same as training)
            u_center = 0.0
            u_range = 1.0
            pred_u = u_center + u_range * torch.tanh(pred_u_raw)
            
            # Translate and softmax for delta
            last = pred_delta_raw[-1]
            pred_delta_translated = pred_delta_raw - last
            pred_delta = F.softmax(pred_delta_translated, dim=-1) * config.TIME_HORIZON
            
            # Reshape
            pred_u = pred_u.reshape(config.N_PHASES, config.N_CONTROL_INPUTS)
            
            # Compute cost
            x0_sys = x0_tensor.to(network.sys.device).to(network.sys.dtype)
            cost_nn = evaluate_cost_functional(network.sys, pred_u, pred_delta, x0_sys).item()
            
            # Compare with optimal
            u_opt = torch.tensor(opt_sol['u'], device=device)
            delta_opt = torch.tensor(opt_sol['delta'], device=device)
            cost_opt = opt_sol['cost']
            
            # Compute errors
            u_error = torch.norm(pred_u - u_opt).item()
            delta_error = torch.norm(pred_delta - delta_opt).item()
            u_mse = torch.mean((pred_u - u_opt)**2).item()
            delta_mse = torch.mean((pred_delta - delta_opt)**2).item()
            
            results['costs_nn'].append(cost_nn)
            results['costs_opt'].append(cost_opt)
            results['u_errors'].append(u_error)
            results['delta_errors'].append(delta_error)
            results['cost_diff'].append(np.abs(cost_nn - cost_opt))
            results['u_mse'].append(u_mse)
            results['delta_mse'].append(delta_mse)
    
    # Compute statistics
    stats = {
        'mean_cost_nn': np.mean(results['costs_nn']),
        'std_cost_nn': np.std(results['costs_nn']),
        'mean_cost_opt': np.mean(results['costs_opt']),
        'std_cost_opt': np.std(results['costs_opt']),
        'mean_u_error': np.mean(results['u_errors']),
        'std_u_error': np.std(results['u_errors']),
        'mean_delta_error': np.mean(results['delta_errors']),
        'std_delta_error': np.std(results['delta_errors']),
        'mean_cost_diff': np.nanmean(results['cost_diff']),
        'std_cost_diff': np.nanstd(results['cost_diff']),
        'mean_u_mse': np.mean(results['u_mse']),
        'mean_delta_mse': np.mean(results['delta_mse']),
    }
    
    return {'raw': results, 'stats': stats}


# ============================================================================
# Visualization
# ============================================================================

def plot_architecture_comparison(results_df: pd.DataFrame, config: ExperimentConfig):
    """Create comprehensive comparison plots."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # 1. Cost vs Number of Parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Mean cost vs n_params
    ax = axes[0, 0]
    ax.scatter(results_df['n_params'], results_df['mean_cost_nn'], alpha=0.6, s=100)
    ax.axhline(results_df['mean_cost_opt'].mean(), color='r', linestyle='--', label='Optimal (mean)')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Mean Cost (NN)', fontsize=12)
    ax.set_title('Cost vs Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cost difference vs n_params
    ax = axes[0, 1]
    ax.scatter(results_df['n_params'], results_df['mean_cost_diff'], alpha=0.6, s=100, c=results_df['n_hidden_layers'], cmap='viridis')
    ax.axhline(0.0, color='r', linestyle='--', label='Optimal difference')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Cost Difference |NN - Opt|', fontsize=12)
    ax.set_title('Cost Difference vs Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: U error vs n_params
    ax = axes[0, 2]
    ax.scatter(results_df['n_params'], results_df['mean_u_mse'], alpha=0.6, s=100)
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Mean U MSE', fontsize=12)
    ax.set_title('Control Error vs Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Delta error vs n_params
    ax = axes[1, 0]
    ax.scatter(results_df['n_params'], results_df['mean_delta_mse'], alpha=0.6, s=100)
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Mean Delta MSE', fontsize=12)
    ax.set_title('Duration Error vs Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Cost vs n_hidden_layers
    ax = axes[1, 1]
    grouped = results_df.groupby('n_hidden_layers').agg({
        'mean_cost_nn': ['mean', 'std'],
        'mean_cost_opt': 'mean'
    })
    x = grouped.index
    y = grouped[('mean_cost_nn', 'mean')]
    yerr = grouped[('mean_cost_nn', 'std')]
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, markersize=8, linewidth=2)
    ax.axhline(grouped[('mean_cost_opt', 'mean')].mean(), color='r', linestyle='--', label='Optimal')
    ax.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax.set_ylabel('Mean Cost (NN)', fontsize=12)
    ax.set_title('Cost vs Network Depth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Training time vs n_params
    ax = axes[1, 2]
    ax.scatter(results_df['n_params'], results_df['training_time'], alpha=0.6, s=100, c=results_df['final_train_loss'], cmap='coolwarm')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Training Time (s)', fontsize=12)
    ax.set_title('Training Time vs Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    plt.colorbar(ax.collections[0], ax=ax, label='Final Train Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'architecture_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config.PLOTS_DIR, 'architecture_comparison.pdf'), bbox_inches='tight')
    print(f"Saved comparison plots to {config.PLOTS_DIR}")
    plt.show()
    
    # 2. Detailed per-architecture comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sort by n_params for better visualization
    results_sorted = results_df.sort_values('n_params')
    x_pos = np.arange(len(results_sorted))
    
    # Cost comparison
    ax = axes[0]
    width = 0.35
    ax.bar(x_pos - width/2, results_sorted['mean_cost_nn'], width, label='NN', alpha=0.8)
    ax.bar(x_pos + width/2, results_sorted['mean_cost_opt'], width, label='Optimal', alpha=0.8)
    ax.set_xlabel('Architecture (sorted by params)', fontsize=11)
    ax.set_ylabel('Mean Cost', fontsize=11)
    ax.set_title('Cost Comparison by Architecture', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_sorted['name'], rotation=90, fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error comparison
    ax = axes[1]
    ax.bar(x_pos, results_sorted['mean_u_mse'], alpha=0.8, label='U MSE')
    ax.set_xlabel('Architecture (sorted by params)', fontsize=11)
    ax.set_ylabel('Mean U MSE', fontsize=11)
    ax.set_title('Control Error by Architecture', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_sorted['name'], rotation=90, fontsize=7)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Performance summary
    ax = axes[2]
    performance_score = (
        results_sorted['mean_cost_diff'] * 0.5 +  # Cost difference (50% weight)
        (results_sorted['mean_u_mse'] / results_sorted['mean_u_mse'].max()) * 0.3 +  # U error (30%)
        (results_sorted['mean_delta_mse'] / results_sorted['mean_delta_mse'].max()) * 0.2  # Delta error (20%)
    )
    colors = plt.cm.RdYlGn_r(performance_score / performance_score.max())
    ax.bar(x_pos, 1.0 / performance_score, color=colors, alpha=0.8)
    ax.set_xlabel('Architecture (sorted by params)', fontsize=11)
    ax.set_ylabel('Performance Score (higher is better)', fontsize=11)
    ax.set_title('Overall Performance by Architecture', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_sorted['name'], rotation=90, fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'detailed_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved detailed comparison to {config.PLOTS_DIR}")
    plt.show()


def plot_pareto_front(results_df: pd.DataFrame, config: ExperimentConfig):
    """Plot Pareto front: performance vs computational cost."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pareto 1: Cost Difference vs n_params
    ax = axes[0]
    scatter = ax.scatter(
        results_df['n_params'],
        results_df['mean_cost_diff'],
        s=results_df['training_time']*2,  # Size by training time
        c=results_df['n_hidden_layers'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.axhline(0.0, color='r', linestyle='--', linewidth=2, label='Optimal')
    ax.set_xlabel('Number of Parameters (Computational Cost)', fontsize=12)
    ax.set_ylabel('Cost Difference (NN - Optimal)', fontsize=12)
    ax.set_title('Pareto Front: Performance vs Model Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Number of Hidden Layers')
    
    # Pareto 2: Combined error vs training time
    ax = axes[1]
    combined_error = results_df['mean_u_mse'] + results_df['mean_delta_mse']
    scatter = ax.scatter(
        results_df['training_time'],
        combined_error,
        s=results_df['n_params']/100,  # Size by n_params
        c=results_df['mean_cost_diff'],
        cmap='RdYlGn_r',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Training Time (s)', fontsize=12)
    ax.set_ylabel('Combined Error (U MSE + Delta MSE)', fontsize=12)
    ax.set_title('Pareto Front: Error vs Training Time', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cost Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'pareto_front.png'), dpi=300, bbox_inches='tight')
    print(f"Saved Pareto front to {config.PLOTS_DIR}")
    plt.show()


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    """Run the complete architecture comparison study."""
    
    print("="*70)
    print("ARCHITECTURE COMPARISON STUDY")
    print("="*70)
    
    # Setup
    config = ExperimentConfig()
    config.create_directories()
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for this experiment.")
        return
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"Experiment directory: {config.EXPERIMENT_DIR}")
    
    # Generate architectures
    n_nn_outputs = config.N_PHASES * (config.N_CONTROL_INPUTS + 1)
    architectures = generate_architectures(config.N_NN_INPUTS, n_nn_outputs)
    
    print(f"\nGenerated {len(architectures)} architectures to test")
    
    # Generate data
    print("\nGenerating training, validation, and test data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    X_train = torch.empty(config.N_SAMPLES_TRAIN, config.N_NN_INPUTS).uniform_(config.X_MIN, config.X_MAX)
    X_val = torch.empty(config.N_SAMPLES_VAL, config.N_NN_INPUTS).uniform_(config.X_MIN, config.X_MAX)
    X_test = torch.empty(config.N_SAMPLES_TEST, config.N_NN_INPUTS).uniform_(config.X_MIN, config.X_MAX)
    
    # Compute optimal solutions for test set
    print(f"\nComputing optimal solutions for {config.N_SAMPLES_TEST} test samples...")
    optimal_solutions = []
    for i, x0_tensor in enumerate(X_test):
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{config.N_SAMPLES_TEST}")
        x0 = x0_tensor.numpy()
        opt_sol = compute_optimal_solution(x0, config)
        optimal_solutions.append(opt_sol)
    
    print("✓ Optimal solutions computed")
    
    # Train all architectures
    print(f"\n{'='*70}")
    print("TRAINING PHASE")
    print(f"{'='*70}")
    
    results = []
    trained_networks = {}  # Store trained networks for later use
    
    for idx, arch in enumerate(architectures):
        print(f"\n[{idx+1}/{len(architectures)}] ", end='')
        
        try:
            # Train
            network, history = train_architecture(arch, X_train, X_val, config, verbose=True)
            
            # Evaluate on test set
            print("  Evaluating on test set...")
            eval_results = evaluate_network_on_test_set(network, X_test, optimal_solutions, config)
            
            # Store results
            result = {
                'name': arch['name'],
                'layer_sizes': str(arch['layer_sizes']),
                'n_hidden_layers': arch['n_hidden_layers'],
                'n_params': arch['n_params'],
                'training_time': history['training_time'],
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                **eval_results['stats']
            }
            results.append(result)
            
            # Store trained network
            trained_networks[arch['name']] = network
            
            print(f"  ✓ Cost Difference: {eval_results['stats']['mean_cost_diff']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(config.RESULTS_DIR, 'architecture_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Save detailed results as JSON
    results_json_path = os.path.join(config.RESULTS_DIR, 'architecture_comparison_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to {results_json_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    print(results_df.describe())
    
    print(f"\n{'='*70}")
    print("BEST ARCHITECTURES")
    print(f"{'='*70}\n")
    
    # Best by cost difference
    best_cost = results_df.loc[results_df['mean_cost_diff'].idxmin()]
    print(f"Best Cost Difference: {best_cost['name']}")
    print(f"  Cost Difference: {best_cost['mean_cost_diff']:.4f}")
    print(f"  Parameters: {best_cost['n_params']:,}")
    print(f"  Training Time: {best_cost['training_time']:.2f}s\n")
    
    # Best by U error
    best_u = results_df.loc[results_df['mean_u_mse'].idxmin()]
    print(f"Best U Error: {best_u['name']}")
    print(f"  U MSE: {best_u['mean_u_mse']:.6f}")
    print(f"  Parameters: {best_u['n_params']:,}")
    print(f"  Training Time: {best_u['training_time']:.2f}s\n")
    
    # Best efficiency (cost difference / n_params)
    results_df['efficiency'] = 1.0 / (results_df['mean_cost_diff'] * np.log10(results_df['n_params']))
    best_eff = results_df.loc[results_df['efficiency'].idxmax()]
    print(f"Most Efficient: {best_eff['name']}")
    print(f"  Efficiency Score: {best_eff['efficiency']:.4f}")
    print(f"  Cost Difference: {best_eff['mean_cost_diff']:.4f}")
    print(f"  Parameters: {best_eff['n_params']:,}")
    print(f"  Training Time: {best_eff['training_time']:.2f}s\n")
    
    # Visualization
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}\n")
    
    # plot_architecture_comparison(results_df, config)
    # plot_pareto_front(results_df, config)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    print(f"Models saved to: {config.MODELS_DIR}")
    
    # Run simulation with top 5 models
    print(f"\n{'='*70}")
    print("SIMULATION WITH TOP 5 MODELS")
    print(f"{'='*70}\n")
    
    run_simulation_comparison(results_df, trained_networks, X_test, optimal_solutions, config)

    # Run batch statistics on unseen test set
    print(f"\n{'='*70}")
    print("BATCH STATISTICS ON UNSEEN X0 VALUES")
    print(f"{'='*70}\n")
    run_batch_simulation(results_df, trained_networks, X_test, optimal_solutions, config, top_k=5)


# ============================================================================
# Simulation and Visualization
# ============================================================================

def propagate_state_fine_grid(sys: SwiLin, u_all: np.ndarray, delta_all: np.ndarray, 
                               x0: np.ndarray, n_steps_per_phase: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate the state using a fine time grid for smooth visualization.
    
    Args:
        sys: SwiLin system
        u_all: Control inputs (n_phases, n_inputs)
        delta_all: Phase durations (n_phases,)
        x0: Initial state (n_states,)
        n_steps_per_phase: Number of time steps per phase
        
    Returns:
        t_fine: Fine time grid
        x_fine: State trajectory on fine grid
    """
    n_phases = len(delta_all)
    n_states = len(x0)
    
    # Build fine time grid
    t_fine_list = []
    x_fine_list = [x0.reshape(-1)]
    
    t_current = 0.0
    x_current = x0.reshape(-1, 1)
    
    for phase_idx in range(n_phases):
        # Get phase parameters
        mode_idx = phase_idx % len(sys.A)
        A = sys.A[mode_idx].cpu().numpy() if torch.is_tensor(sys.A[mode_idx]) else sys.A[mode_idx]
        B = sys.B[mode_idx].cpu().numpy() if torch.is_tensor(sys.B[mode_idx]) else sys.B[mode_idx]
        u_phase = u_all[phase_idx].reshape(-1, 1)
        delta_phase = delta_all[phase_idx]
        
        # Fine time grid for this phase
        t_phase = np.linspace(0, delta_phase, n_steps_per_phase)
        
        for dt in t_phase[1:]:  # Skip first point (already have it)
            # State propagation: x(t+dt) = exp(A*dt)*x(t) + int_0^dt exp(A*s)*B*u ds
            # For piecewise constant u: x(t+dt) = exp(A*dt)*x(t) + A^{-1}(exp(A*dt)-I)*B*u
            eAdt = scipy.linalg.expm(A * dt)
            
            if np.linalg.matrix_rank(A) == A.shape[0]:  # A is invertible
                Ainv = np.linalg.inv(A)
                integral_term = Ainv @ (eAdt - np.eye(A.shape[0])) @ B @ u_phase
            else:
                # Use numerical integration if A is singular
                integral_term = dt * B @ u_phase
            
            x_next = eAdt @ x_current + integral_term
            x_current = x_next
            
            t_fine_list.append(t_current + dt)
            x_fine_list.append(x_current.flatten())
        
        t_current += delta_phase
        x_current = x_fine_list[-1].reshape(-1, 1)
    
    # Build final arrays (x_fine_list already starts with x0)
    t_fine = np.array([0.0] + t_fine_list)
    x_fine = np.array(x_fine_list)
    
    return t_fine, x_fine


def compute_trajectory_error(t_ref: np.ndarray, x_ref: np.ndarray, 
                             t_other: np.ndarray, x_other: np.ndarray) -> float:
    """
    Compute RMSE between two trajectories by interpolating the "other" trajectory
    onto the reference time grid.

    Args:
        t_ref: reference time grid (1D array)
        x_ref: reference states (len(t_ref), n_states)
        t_other: other time grid
        x_other: other states (len(t_other), n_states)

    Returns:
        rmse: root-mean-square error over time and state dimensions
    """
    # Ensure arrays are numpy
    t_ref = np.asarray(t_ref)
    t_other = np.asarray(t_other)
    x_ref = np.asarray(x_ref)
    x_other = np.asarray(x_other)

    # Number of state dimensions
    if x_ref.ndim == 1:
        x_ref = x_ref.reshape(-1, 1)
    if x_other.ndim == 1:
        x_other = x_other.reshape(-1, 1)
    n_states = x_ref.shape[1]

    # Interpolate each state dimension of x_other onto t_ref
    x_other_interp = np.zeros_like(x_ref)
    for j in range(n_states):
        xj_other = x_other[:, j]
        # Use numpy.interp (1D) which will clip/extrapolate at edges
        x_other_interp[:, j] = np.interp(t_ref, t_other, xj_other)

    # Compute RMSE over time and states
    err = x_other_interp - x_ref
    mse = np.mean(err**2)
    rmse = float(np.sqrt(mse))
    return rmse


def plot_simulation_results(results_dict: Dict, config: ExperimentConfig, save_name: str = 'simulation_comparison.png'):
    """
    Plot simulation results comparing top models and optimal solution.
    
    Args:
        results_dict: Dictionary with 'name', 'u', 'delta', 'x0', 't_fine', 'x_fine' for each method
        config: Configuration
        save_name: Filename for saving plot
    """
    n_methods = len(results_dict)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color map for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    # Plot 1: State trajectories
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (method_name, data) in enumerate(results_dict.items()):
        if method_name == 'Optimal':
            ax1.plot(data['t_fine'], data['x_fine'][:, 0], 
                    linewidth=3, alpha=0.9, label=method_name, color='black', linestyle='--')
        else:
            ax1.plot(data['t_fine'], data['x_fine'][:, 0], 
                    linewidth=2, alpha=0.7, label=method_name, color=colors[idx])
        
        # Add vertical lines for switches (only for first method to avoid clutter)
        if idx == 0:
            switch_times = np.cumsum(data['delta'])
            for t_switch in switch_times[:-1]:  # Exclude final time
                ax1.axvline(t_switch, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('State x(t)', fontsize=12)
    ax1.set_title('State Trajectories (vertical lines show mode switches)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, config.TIME_HORIZON)
    ax1.set_ylim(config.X_MIN*3.0, config.X_MAX*3.0)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control inputs (piecewise constant)
    ax2 = fig.add_subplot(gs[1, :])
    for idx, (method_name, data) in enumerate(results_dict.items()):
        u = data['u'].flatten()
        delta = data['delta']
        
        # Create piecewise constant signal
        times = np.concatenate([[0.0], np.cumsum(delta)])
        times_stairs = np.repeat(times, 2)[1:-1]
        u_stairs = np.repeat(u, 2)
        
        if method_name == 'Optimal':
            ax2.plot(times_stairs, u_stairs, linewidth=3, alpha=0.9, 
                    label=method_name, color='black', linestyle='--', drawstyle='steps-post')
        else:
            ax2.plot(times_stairs, u_stairs, linewidth=2, alpha=0.7, 
                    label=method_name, color=colors[idx], drawstyle='steps-post')
        
        # Add vertical lines for switches
        if idx == 0:
            for t_switch in times[1:-1]:
                ax2.axvline(t_switch, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Control u(t)', fontsize=12)
    ax2.set_title('Piecewise Constant Control Signals', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, config.TIME_HORIZON)
    
    # Plot 3: Phase durations comparison
    ax3 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(config.N_PHASES)
    width = 0.8 / n_methods
    
    for idx, (method_name, data) in enumerate(results_dict.items()):
        offset = (idx - n_methods/2) * width + width/2
        if method_name == 'Optimal':
            ax3.bar(x_pos + offset, data['delta'], width, 
                   label=method_name, alpha=0.9, color='black', edgecolor='white', linewidth=1.5)
        else:
            ax3.bar(x_pos + offset, data['delta'], width, 
                   label=method_name, alpha=0.7, color=colors[idx], edgecolor='white', linewidth=0.5)
    
    ax3.set_xlabel('Phase Index', fontsize=12)
    ax3.set_ylabel('Duration δ', fontsize=12)
    ax3.set_title('Phase Durations', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Control comparison
    ax4 = fig.add_subplot(gs[2, 1])
    for idx, (method_name, data) in enumerate(results_dict.items()):
        offset = (idx - n_methods/2) * width + width/2
        if method_name == 'Optimal':
            ax4.bar(x_pos + offset, data['u'].flatten(), width, 
                   label=method_name, alpha=0.9, color='black', edgecolor='white', linewidth=1.5)
        else:
            ax4.bar(x_pos + offset, data['u'].flatten(), width, 
                   label=method_name, alpha=0.7, color=colors[idx], edgecolor='white', linewidth=0.5)
    
    ax4.set_xlabel('Phase Index', fontsize=12)
    ax4.set_ylabel('Control u', fontsize=12)
    ax4.set_title('Control Values per Phase', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Simulation Comparison: Top 5 Models vs Optimal Solution\n(x0 = {data["x0"]})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    save_path = os.path.join(config.PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved simulation plot to {save_path}")
    plt.show()
    
def create_switched_system(n_phases: int, time_horizon: float) -> SwiLin:
    """
    Create a SwiLin system with predefined A, B matrices for simulation.
    """
    model = {
        # 'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
        # 'B': [np.array([[0.25], [2], [0]])],
        'A': [np.array([[1]]), np.array([[-3]])],
        'B': [np.array([[2]]), np.array([[-1]])],
    }


    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    # xr = np.array([1, -3])
    
    swi_lin = SwiLin(
        n_phases, 
        n_states,
        n_inputs,
        time_horizon, 
        auto=False, 
    )
    
    # Load model
    swi_lin.load_model(model)

    Q = 1. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    E = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin.load_weights(Q, R, E)
    
    return swi_lin


def run_simulation_comparison(results_df: pd.DataFrame, trained_networks: Dict[str, SwiLinNN],
                               X_test: torch.Tensor, optimal_solutions: List[Dict], 
                               config: ExperimentConfig):
    """
    Run simulation comparison for top 5 models and optimal solution.
    """
    # Select top 5 architectures by cost difference
    top5 = results_df.nsmallest(5, 'mean_cost_diff')
    
    print("Top 5 architectures selected:")
    for i, row in enumerate(top5.itertuples(), 1):
        print(f"  {i}. {row.name} - Cost Diff: {row.mean_cost_diff:.4f}")
    
    # Select a test case (use first test sample)
    test_idx = 0
    x0_test = X_test[test_idx].cpu().numpy()
    opt_sol = optimal_solutions[test_idx]
    
    print(f"\nRunning simulation for x0 = {x0_test}")
    
    # Create results dictionary
    results_dict = {}
    
    # Add optimal solution
    sys_opt = create_switched_system(config.N_PHASES, config.TIME_HORIZON)
    t_fine_opt, x_fine_opt = propagate_state_fine_grid(
        sys_opt, 
        opt_sol['u'], 
        opt_sol['delta'], 
        x0_test,
        n_steps_per_phase=100
    )
    
    results_dict['Optimal'] = {
        'u': opt_sol['u'],
        'delta': opt_sol['delta'],
        'x0': x0_test,
        't_fine': t_fine_opt,
        'x_fine': x_fine_opt,
        'cost': opt_sol['cost']
    }
    
    # Add top 5 models
    for i, row in enumerate(top5.itertuples(), 1):
        # Get the trained network from dictionary
        if row.name not in trained_networks:
            print(f"  Warning: Trained network not found for {row.name}")
            continue
        
        network = trained_networks[row.name]
        network.eval()
        network.to(config.DEVICE)
        
        # Get prediction
        with torch.no_grad():
            x0_input = torch.tensor(x0_test, dtype=network.sys.dtype, device=config.DEVICE).unsqueeze(0)
            output = network(x0_input).squeeze(0)
            
            # Transform outputs
            n_control_outputs = config.N_PHASES * config.N_CONTROL_INPUTS
            u_raw = output[:n_control_outputs]
            delta_raw = output[n_control_outputs:]
            
            # Apply transformations
            u = torch.tanh(u_raw).reshape(config.N_PHASES, config.N_CONTROL_INPUTS)
            delta = F.softmax(delta_raw - delta_raw[-1], dim=-1) * config.TIME_HORIZON
            
            u_np = u.cpu().numpy()
            delta_np = delta.cpu().numpy()
        
        # Propagate state
        t_fine, x_fine = propagate_state_fine_grid(
            network.sys,
            u_np,
            delta_np,
            x0_test,
            n_steps_per_phase=100
        )
        
        # Compute cost
        x0_sys = torch.tensor(x0_test, dtype=network.sys.dtype, device=network.sys.device)
        cost_nn = evaluate_cost_functional(network.sys, 
                                          torch.tensor(u_np, dtype=network.sys.dtype, device=network.sys.device),
                                          torch.tensor(delta_np, dtype=network.sys.dtype, device=network.sys.device),
                                          x0_sys).item()
        
        results_dict[row.name] = {
            'u': u_np,
            'delta': delta_np,
            'x0': x0_test,
            't_fine': t_fine,
            'x_fine': x_fine,
            'cost': cost_nn
        }
        
        print(f"  {i}. {row.name}: Cost = {cost_nn:.6f} (Optimal = {opt_sol['cost']:.6f})")
    
    # Plot results
    print("\nGenerating simulation plots...")
    plot_simulation_results(results_dict, config)
    
    # Print cost comparison
    print(f"\n{'='*70}")
    print("COST COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<30s} {'Cost':>12s} {'Cost - OCP':>15s} {'Traj RMSE':>12s}")
    print("-"*90)
    
    opt_cost = results_dict['Optimal']['cost']
    # Compute trajectory RMSEs w.r.t. optimal and print table
    # Ensure optimal trajectory exists
    t_ref = results_dict['Optimal']['t_fine']
    x_ref = results_dict['Optimal']['x_fine']
    # Add RMSE for optimal (zero)
    results_dict['Optimal']['traj_rmse'] = 0.0

    for method_name, data in results_dict.items():
        cost = data['cost']
        diff = cost - opt_cost
        if method_name != 'Optimal':
            try:
                rmse = compute_trajectory_error(t_ref, x_ref, data['t_fine'], data['x_fine'])
            except Exception:
                rmse = float('nan')
            results_dict[method_name]['traj_rmse'] = rmse
        else:
            rmse = 0.0

        print(f"{method_name:<30s} {cost:12.6f} {diff:+15.6f} {rmse:12.6f}")


def run_batch_simulation(results_df: pd.DataFrame, trained_networks: Dict[str, SwiLinNN],
                         X_eval: torch.Tensor, optimal_solutions: List[Dict],
                         config: ExperimentConfig, top_k: int = 5):
    """
    Run simulations over a batch of initial states and compute statistics (mean, std)
    for cost and trajectory RMSE for the Optimal solution and the top-k networks.
    """
    topk = results_df.nsmallest(top_k, 'mean_cost_diff')
    method_names = ['Optimal'] + list(topk['name'])

    # Initialize storage
    stats = {m: {'costs': [], 'rmses': []} for m in method_names}

    # Pre-create switched system for optimal propagation
    sys_opt = create_switched_system(config.N_PHASES, config.TIME_HORIZON)

    for i, x0_tensor in enumerate(X_eval):
        x0 = x0_tensor.cpu().numpy()
        opt_sol = optimal_solutions[i]

        # Optimal trajectory
        try:
            t_ref, x_ref = propagate_state_fine_grid(sys_opt, opt_sol['u'], opt_sol['delta'], x0, n_steps_per_phase=50)
        except Exception:
            # Skip this sample if optimal propagation fails
            continue

        stats['Optimal']['costs'].append(opt_sol['cost'])
        stats['Optimal']['rmses'].append(0.0)

        # Evaluate each top-k network
        for name in topk['name']:
            if name not in trained_networks:
                stats[name]['costs'].append(float('nan'))
                stats[name]['rmses'].append(float('nan'))
                continue

            network = trained_networks[name]
            network.eval()
            network.to(config.DEVICE)

            with torch.no_grad():
                x0_input = torch.tensor(x0, dtype=network.sys.dtype, device=config.DEVICE).unsqueeze(0)
                output = network(x0_input).squeeze(0)
                n_control_outputs = config.N_PHASES * config.N_CONTROL_INPUTS
                u_raw = output[:n_control_outputs]
                delta_raw = output[n_control_outputs:]
                u = torch.tanh(u_raw).reshape(config.N_PHASES, config.N_CONTROL_INPUTS)
                delta = F.softmax(delta_raw - delta_raw[-1], dim=-1) * config.TIME_HORIZON
                u_np = u.cpu().numpy()
                delta_np = delta.cpu().numpy()

            # Propagate and evaluate
            try:
                t_other, x_other = propagate_state_fine_grid(network.sys, u_np, delta_np, x0, n_steps_per_phase=50)
                cost_nn = evaluate_cost_functional(network.sys,
                                                  torch.tensor(u_np, dtype=network.sys.dtype, device=network.sys.device),
                                                  torch.tensor(delta_np, dtype=network.sys.dtype, device=network.sys.device),
                                                  torch.tensor(x0, dtype=network.sys.dtype, device=network.sys.device)).item()
                rmse = compute_trajectory_error(t_ref, x_ref, t_other, x_other)
            except Exception:
                cost_nn = float('nan')
                rmse = float('nan')

            stats[name]['costs'].append(cost_nn)
            stats[name]['rmses'].append(rmse)

    # Aggregate statistics
    rows = []
    for m in method_names:
        costs = np.array(stats[m]['costs'], dtype=float)
        rmses = np.array(stats[m]['rmses'], dtype=float)
        rows.append({
            'method': m,
            'n_samples': int(np.sum(~np.isnan(costs))) if costs.size else 0,
            'mean_cost': float(np.nanmean(costs)) if costs.size else float('nan'),
            'std_cost': float(np.nanstd(costs)) if costs.size else float('nan'),
            'mean_rmse': float(np.nanmean(rmses)) if rmses.size else float('nan'),
            'std_rmse': float(np.nanstd(rmses)) if rmses.size else float('nan'),
        })

    stats_df = pd.DataFrame(rows).set_index('method')
    print("\nBatch simulation statistics (per-method):")
    print(stats_df)

    # Save to CSV
    out_path = os.path.join(config.RESULTS_DIR, f'batch_simulation_stats_top{top_k}.csv')
    stats_df.to_csv(out_path)
    print(f"Saved batch statistics to {out_path}")


if __name__ == "__main__":
    main()
