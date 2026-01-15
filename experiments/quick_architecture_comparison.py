"""
Quick Architecture Comparison Study

A lightweight version of the architecture comparison that runs faster for initial testing.
Tests fewer architectures with fewer epochs and samples.
"""

import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from ocslc.switched_linear_mpc import SwitchedLinearMPC as SwiLin_casadi

from src.switched_linear_torch import SwiLin
from src.training import SwiLinNN, train_neural_network, evaluate_cost_functional


# ============================================================================
# Quick Configuration
# ============================================================================

class QuickConfig:
    """Minimal configuration for fast testing."""
    
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
    
    # Reduced training setup for speed
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SAMPLES_TRAIN = 500  # Reduced from 1000
    N_SAMPLES_VAL = 50     # Reduced from 100
    N_SAMPLES_TEST = 20    # Reduced from 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 50          # Reduced from 200
    BATCH_SIZE = N_SAMPLES_TRAIN
    OPTIMIZER = 'adam'
    
    # Data range
    X_MIN = -5.0
    X_MAX = 5.0
    
    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 15  # Reduced from 30
    
    # Save paths
    EXPERIMENT_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'quick_study')
    RESULTS_DIR = os.path.join(EXPERIMENT_DIR, 'results')
    PLOTS_DIR = os.path.join(EXPERIMENT_DIR, 'plots')
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.RESULTS_DIR, cls.PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)


def generate_quick_architectures(n_inputs: int, n_outputs: int):
    """Generate a small set of representative architectures."""
    return [
        # Small
        {
            'name': 'small_1x16',
            'layer_sizes': [n_inputs, 16, n_outputs],
            'n_hidden_layers': 1,
        },
        # Medium depth
        {
            'name': 'medium_2x32',
            'layer_sizes': [n_inputs, 32, 32, n_outputs],
            'n_hidden_layers': 2,
        },
        # Medium wide
        {
            'name': 'medium_2x64',
            'layer_sizes': [n_inputs, 64, 64, n_outputs],
            'n_hidden_layers': 2,
        },
        # Deep narrow
        {
            'name': 'deep_4x32',
            'layer_sizes': [n_inputs, 32, 32, 32, 32, n_outputs],
            'n_hidden_layers': 4,
        },
        # Wide shallow
        {
            'name': 'wide_1x128',
            'layer_sizes': [n_inputs, 128, n_outputs],
            'n_hidden_layers': 1,
        },
        # Pyramidal
        {
            'name': 'pyramid_128_64_32',
            'layer_sizes': [n_inputs, 128, 64, 32, n_outputs],
            'n_hidden_layers': 3,
        },
        # Large
        {
            'name': 'large_3x128',
            'layer_sizes': [n_inputs, 128, 128, 128, n_outputs],
            'n_hidden_layers': 3,
        },
    ]


def count_parameters(network):
    """Count total trainable parameters."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def compute_optimal_solution_simple(x0, config):
    """Simplified optimal solution computation using CasADi."""
    # Create system
    model = {
        'A': [np.array([[1]]), np.array([[-3]])],
        'B': [np.array([[2]]), np.array([[-1]])],
    }
    
    from scipy.linalg import solve_continuous_are
    
    model = {
    'A': [np.array([[1]]), np.array([[-3]])],
    'B': [np.array([[2]]), np.array([[-1]])],
}

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    # x0 = np.array([1.3440, -4.5850, 5.6470])

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

    Q = 1. * np.eye(n_states)
    R = 1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
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
    
    u, delta, states_ = swi_lin_casadi.solve()
    cost = swi_lin_casadi.opt_cost
    
    return {
        'u': np.array(u),
        'delta': np.array(delta),
        'cost': cost.item(),
    }


def evaluate_network(network, X_test, optimal_solutions, config):
    """Evaluate network performance."""
    network.eval()
    results = {
        'cost_nn': [],
        'cost_opt': [],
        'cost_diff': [],
        'u_mse': [],
        'delta_mse': [],
    }
    
    with torch.no_grad():
        for i, x0 in enumerate(X_test):
            # Network prediction
            x0_input = x0.to(config.DEVICE).unsqueeze(0)
            output = network(x0_input).squeeze(0)
            
            # Transform outputs
            n_u = config.N_PHASES * config.N_CONTROL_INPUTS
            u_raw = output[:n_u]
            delta_raw = output[n_u:]
            
            u = torch.tanh(u_raw).reshape(config.N_PHASES, config.N_CONTROL_INPUTS)
            delta = F.softmax(delta_raw - delta_raw[-1], dim=-1) * config.TIME_HORIZON
            
            # Compute cost
            x0_sys = x0.to(network.sys.device).to(network.sys.dtype)
            cost_nn = evaluate_cost_functional(network.sys, u, delta, x0_sys).item()
            
            # Compare with optimal
            opt = optimal_solutions[i]
            u_opt = torch.tensor(opt['u'], device=config.DEVICE)
            delta_opt = torch.tensor(opt['delta'], device=config.DEVICE)
            
            results['cost_nn'].append(cost_nn)
            results['cost_opt'].append(opt['cost'])
            results['cost_diff'].append(np.abs(cost_nn - opt['cost']))
            results['u_mse'].append(torch.mean((u - u_opt)**2).item())
            results['delta_mse'].append(torch.mean((delta - delta_opt)**2).item())
    
    return {
        'mean_cost_nn': np.mean(results['cost_nn']),
        'mean_cost_opt': np.mean(results['cost_opt']),
        'mean_cost_diff': np.mean(results['cost_diff']),
        'mean_u_mse': np.mean(results['u_mse']),
        'mean_delta_mse': np.mean(results['delta_mse']),
    }


def plot_quick_results(results_df, config):
    """Generate quick comparison plots."""
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Cost difference vs params
    ax = axes[0, 0]
    ax.scatter(results_df['n_params'], results_df['mean_cost_diff'], s=100, alpha=0.6)
    ax.axhline(0.0, color='r', linestyle='--', label='Optimal')
    for idx, row in results_df.iterrows():
        ax.annotate(row['name'], (row['n_params'], row['mean_cost_diff']),
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Cost Difference |NN - Optimal|')
    ax.set_title('Performance vs Model Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. U error
    ax = axes[0, 1]
    ax.bar(range(len(results_df)), results_df['mean_u_mse'], alpha=0.8)
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Mean U MSE')
    ax.set_title('Control Input Error')
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['name'], rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Training time
    ax = axes[1, 0]
    colors = plt.cm.RdYlGn_r(results_df['mean_cost_diff'] / results_df['mean_cost_diff'].max())
    ax.bar(range(len(results_df)), results_df['training_time'], color=colors, alpha=0.8)
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time (colored by cost difference)')
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['name'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    summary = results_df[['name', 'n_params', 'mean_cost_diff', 'training_time']].copy()
    summary['n_params'] = summary['n_params'].apply(lambda x: f"{x:,}")
    summary['mean_cost_diff'] = summary['mean_cost_diff'].apply(lambda x: f"{x:.3f}")
    summary['training_time'] = summary['training_time'].apply(lambda x: f"{x:.1f}s")
    summary.columns = ['Architecture', 'Params', 'Cost Difference', 'Time']
    
    table = ax.table(cellText=summary.values, colLabels=summary.columns,
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color code by performance
    for i in range(len(summary)):
        diff = results_df.iloc[i]['mean_cost_diff']
        if diff < 0.05:
            color = '#90EE90'  # light green
        elif diff < 0.10:
            color = '#FFFFE0'  # light yellow
        else:
            color = '#FFB6C1'  # light red
        for j in range(4):
            table[(i+1, j)].set_facecolor(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'quick_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"Saved plot to {config.PLOTS_DIR}/quick_comparison.png")
    plt.show()


def main():
    """Run quick architecture comparison."""
    
    print("="*70)
    print("QUICK ARCHITECTURE COMPARISON STUDY")
    print("="*70)
    
    config = QuickConfig()
    config.create_directories()
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"Training {config.N_EPOCHS} epochs with {config.N_SAMPLES_TRAIN} samples")
    
    # Generate data
    torch.manual_seed(42)
    np.random.seed(42)
    
    X_train = torch.empty(config.N_SAMPLES_TRAIN, config.N_NN_INPUTS).uniform_(config.X_MIN, config.X_MAX)
    X_val = torch.empty(config.N_SAMPLES_VAL, config.N_NN_INPUTS).uniform_(config.X_MIN, config.X_MAX)
    X_test = torch.empty(config.N_SAMPLES_TEST, config.N_NN_INPUTS).uniform_(config.X_MIN, config.X_MAX)
    
    # Compute optimal solutions
    print(f"\nComputing optimal solutions for {config.N_SAMPLES_TEST} test samples...")
    optimal_solutions = []
    for x0 in X_test:
        opt = compute_optimal_solution_simple(
            x0.numpy(),
            config
        )
        optimal_solutions.append(opt)
    
    # Generate architectures
    n_outputs = config.N_PHASES * (config.N_CONTROL_INPUTS + 1)
    architectures = generate_quick_architectures(config.N_NN_INPUTS, n_outputs)
    
    print(f"\nTesting {len(architectures)} architectures:")
    for arch in architectures:
        print(f"  - {arch['name']}: {arch['layer_sizes']}")
    
    # Train and evaluate
    results = []
    
    for idx, arch in enumerate(architectures):
        print(f"\n[{idx+1}/{len(architectures)}] Training {arch['name']}...")
        
        try:
            # Create network
            network = SwiLinNN(
                layer_sizes=arch['layer_sizes'],
                n_phases=config.N_PHASES,
                activation='relu',
                output_activation='linear'
            )
            arch['n_params'] = count_parameters(network)
            
            # Train
            start = time.time()
            _, history = train_neural_network(
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
                early_stopping=config.EARLY_STOPPING,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            )
            training_time = time.time() - start
            
            # Evaluate
            eval_stats = evaluate_network(network, X_test, optimal_solutions, config)
            
            # Store
            results.append({
                'name': arch['name'],
                'layer_sizes': str(arch['layer_sizes']),
                'n_hidden_layers': arch['n_hidden_layers'],
                'n_params': arch['n_params'],
                'training_time': training_time,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                **eval_stats
            })
            
            print(f"  ✓ Params: {arch['n_params']:,}, Cost difference: {eval_stats['mean_cost_diff']:.4f}, Time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(config.RESULTS_DIR, 'quick_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    
    # Best architecture
    best = results_df.loc[results_df['mean_cost_diff'].idxmin()]
    print(f"\n{'='*70}")
    print(f"BEST ARCHITECTURE: {best['name']}")
    print(f"{'='*70}")
    print(f"Cost difference: {best['mean_cost_diff']:.4f}")
    print(f"Parameters: {best['n_params']:,}")
    print(f"Training time: {best['training_time']:.2f}s")
    print(f"U MSE: {best['mean_u_mse']:.6f}")
    
    # Plot
    plot_quick_results(results_df, config)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
