"""
Results Analyzer

This script loads and analyzes results from architecture comparison studies.
It can compare multiple runs, generate custom plots, and export statistics.

Usage:
    python experiments/analyze_results.py --study quick
    python experiments/analyze_results.py --study comprehensive
    python experiments/analyze_results.py --compare run1.csv run2.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(study_type: str = 'quick') -> pd.DataFrame:
    """Load results from a study."""
    base_dir = Path(__file__).parent.parent
    
    if study_type == 'quick':
        csv_path = base_dir / 'experiments' / 'quick_study' / 'results' / 'quick_results.csv'
    elif study_type == 'comprehensive':
        csv_path = base_dir / 'experiments' / 'architecture_study' / 'results' / 'architecture_comparison_results.csv'
    else:
        csv_path = Path(study_type)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} architectures from {csv_path}")
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal architectures: {len(df)}")
    print(f"Parameter range: {df['n_params'].min():,} - {df['n_params'].max():,}")
    print(f"Hidden layers range: {df['n_hidden_layers'].min()} - {df['n_hidden_layers'].max()}")
    
    print(f"\nCost Difference:")
    print(f"  Mean: {df['mean_cost_diff'].mean():.4f}")
    print(f"  Std:  {df['mean_cost_diff'].std():.4f}")
    print(f"  Min:  {df['mean_cost_diff'].min():.4f}")
    print(f"  Max:  {df['mean_cost_diff'].max():.4f}")
    
    print(f"\nControl Error (U MSE):")
    print(f"  Mean: {df['mean_u_mse'].mean():.6f}")
    print(f"  Std:  {df['mean_u_mse'].std():.6f}")
    print(f"  Min:  {df['mean_u_mse'].min():.6f}")
    
    print(f"\nDuration Error (Delta MSE):")
    print(f"  Mean: {df['mean_delta_mse'].mean():.6f}")
    print(f"  Std:  {df['mean_delta_mse'].std():.6f}")
    print(f"  Min:  {df['mean_delta_mse'].min():.6f}")
    
    print(f"\nTraining Time:")
    print(f"  Mean: {df['training_time'].mean():.2f}s")
    print(f"  Total: {df['training_time'].sum()/60:.2f} minutes")


def print_top_architectures(df: pd.DataFrame, n: int = 5):
    """Print top N architectures by different metrics."""
    print("\n" + "="*70)
    print("TOP ARCHITECTURES")
    print("="*70)
    
    # Best cost difference
    print(f"\nTop {n} by Cost Difference (lower is better):")
    top_cost = df.nsmallest(n, 'mean_cost_diff')[['name', 'n_params', 'mean_cost_diff', 'mean_u_mse']]
    for i, row in enumerate(top_cost.itertuples(), 1):
        print(f"  {i}. {row.name:30s} | params: {row.n_params:8,} | diff: {row.mean_cost_diff:.4f} | u_mse: {row.mean_u_mse:.6f}")
    
    # Best U error
    print(f"\nTop {n} by U MSE (lower is better):")
    top_u = df.nsmallest(n, 'mean_u_mse')[['name', 'n_params', 'mean_cost_diff', 'mean_u_mse']]
    for i, row in enumerate(top_u.itertuples(), 1):
        print(f"  {i}. {row.name:30s} | params: {row.n_params:8,} | diff: {row.mean_cost_diff:.4f} | u_mse: {row.mean_u_mse:.6f}")
    
    # Best efficiency (performance per parameter)
    df_temp = df.copy()
    df_temp['efficiency'] = 1.0 / (df_temp['mean_cost_diff'] * np.log10(df_temp['n_params'] + 1))
    print(f"\nTop {n} by Efficiency (higher is better):")
    top_eff = df_temp.nsmallest(n, 'mean_cost_diff')[['name', 'n_params', 'mean_cost_diff', 'efficiency']]
    for i, row in enumerate(top_eff.itertuples(), 1):
        eff_val = 1.0 / (row.mean_cost_diff * np.log10(row.n_params + 1))
        print(f"  {i}. {row.name:30s} | params: {row.n_params:8,} | diff: {row.mean_cost_diff:.4f} | eff: {eff_val:.4f}")
    
    # Fastest training
    print(f"\nTop {n} by Training Speed (faster):")
    top_speed = df.nsmallest(n, 'training_time')[['name', 'training_time', 'mean_cost_diff', 'n_params']]
    for i, row in enumerate(top_speed.itertuples(), 1):
        print(f"  {i}. {row.name:30s} | time: {row.training_time:6.2f}s | diff: {row.mean_cost_diff:.4f} | params: {row.n_params:8,}")


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot correlation between metrics."""
    metrics = ['n_params', 'n_hidden_layers', 'mean_cost_diff', 
               'mean_u_mse', 'mean_delta_mse', 'training_time',
               'final_train_loss', 'final_val_loss']
    
    # Select available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    corr = df[available_metrics].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Architecture Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved correlation matrix to {save_path}")
    
    plt.show()


def plot_depth_vs_width_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """Analyze effect of depth vs width."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parse layer sizes to get average width
    def get_avg_width(layer_sizes_str):
        try:
            sizes = eval(layer_sizes_str)
            if len(sizes) > 2:
                return np.mean(sizes[1:-1])  # Average of hidden layers
            return sizes[1] if len(sizes) > 1 else 0
        except:
            return 0
    
    df['avg_width'] = df['layer_sizes'].apply(get_avg_width)
    
    # Depth effect
    ax = axes[0]
    depth_stats = df.groupby('n_hidden_layers').agg({
        'mean_cost_ratio': ['mean', 'std', 'min'],
        'n_params': 'mean'
    })
    
    x = depth_stats.index
    y = depth_stats[('mean_cost_ratio', 'mean')]
    yerr = depth_stats[('mean_cost_ratio', 'std')]
    
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, markersize=10, linewidth=2, label='Mean ± Std')
    ax.scatter(x, depth_stats[('mean_cost_ratio', 'min')], marker='*', s=200, 
               color='gold', edgecolor='black', linewidth=1, label='Best', zorder=10)
    ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Optimal')
    ax.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax.set_ylabel('Mean Cost Difference', fontsize=12)
    ax.set_title('Effect of Network Depth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Width effect (for fixed depth if enough data)
    ax = axes[1]
    # Try to find a common depth
    common_depth = df['n_hidden_layers'].mode()[0] if len(df['n_hidden_layers'].mode()) > 0 else 3
    df_fixed_depth = df[df['n_hidden_layers'] == common_depth]
    
    if len(df_fixed_depth) >= 3 and df_fixed_depth['avg_width'].nunique() >= 3:
        width_stats = df_fixed_depth.groupby('avg_width').agg({
            'mean_cost_ratio': ['mean', 'std'],
            'n_params': 'mean'
        })
        
        x = width_stats.index
        y = width_stats[('mean_cost_ratio', 'mean')]
        yerr = width_stats[('mean_cost_ratio', 'std')]
        
        ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, markersize=10, linewidth=2)
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Optimal')
        ax.set_xlabel(f'Average Width (depth={common_depth})', fontsize=12)
        ax.set_ylabel('Mean Cost Difference', fontsize=12)
        ax.set_title('Effect of Network Width', fontsize=13, fontweight='bold')
    else:
        # Alternative: scatter all
        scatter = ax.scatter(df['avg_width'], df['mean_cost_diff'], 
                           s=df['n_params']/10, alpha=0.6, c=df['n_hidden_layers'], cmap='viridis')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Optimal')
        ax.set_xlabel('Average Width', fontsize=12)
        ax.set_ylabel('Cost Difference', fontsize=12)
        ax.set_title('Width vs Performance (all depths)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Hidden Layers')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved depth/width analysis to {save_path}")
    
    plt.show()


def compare_studies(study_files: List[str], labels: Optional[List[str]] = None):
    """Compare multiple study results."""
    if labels is None:
        labels = [f"Study {i+1}" for i in range(len(study_files))]
    
    dfs = []
    for file in study_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    print(f"\n{'='*70}")
    print("COMPARING STUDIES")
    print(f"{'='*70}")
    
    for label, df in zip(labels, dfs):
        print(f"\n{label}:")
        print(f"  Architectures: {len(df)}")
        print(f"  Best cost difference: {df['mean_cost_diff'].min():.4f}")
        print(f"  Mean cost difference: {df['mean_cost_diff'].mean():.4f}")
        print(f"  Param range: {df['n_params'].min():,} - {df['n_params'].max():,}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost difference distributions
    ax = axes[0]
    for label, df in zip(labels, dfs):
        ax.hist(df['mean_cost_diff'], alpha=0.6, label=label, bins=15)
    ax.axvline(0.0, color='r', linestyle='--', linewidth=2, label='Optimal')
    ax.set_xlabel('Cost Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Cost Difference Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pareto front comparison
    ax = axes[1]
    for label, df in zip(labels, dfs):
        ax.scatter(df['n_params'], df['mean_cost_diff'], alpha=0.6, s=100, label=label)
    ax.axhline(0.0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Cost Difference', fontsize=12)
    ax.set_title('Pareto Front Comparison', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def export_latex_table(df: pd.DataFrame, output_file: str, n_rows: int = 10):
    """Export top architectures as LaTeX table."""
    top_df = df.nsmallest(n_rows, 'mean_cost_ratio')[
        ['name', 'n_params', 'n_hidden_layers', 'mean_cost_ratio', 'mean_u_mse', 'training_time']
    ].copy()
    
    top_df.columns = ['Architecture', 'Params', 'Layers', 'Cost Difference', 'U MSE', 'Time (s)']
    
    latex_str = top_df.to_latex(index=False, float_format='%.4f', escape=False)
    
    with open(output_file, 'w') as f:
        f.write(latex_str)
    
    print(f"Exported LaTeX table to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze architecture comparison results')
    parser.add_argument('--study', type=str, choices=['quick', 'comprehensive'], 
                       help='Type of study to analyze')
    parser.add_argument('--file', type=str, help='Path to specific results CSV file')
    parser.add_argument('--compare', nargs='+', help='Compare multiple result files')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top architectures to show')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--export-latex', type=str, help='Export LaTeX table to file')
    
    args = parser.parse_args()
    
    # Load results
    if args.compare:
        compare_studies(args.compare)
        return
    
    if args.file:
        df = pd.read_csv(args.file)
    elif args.study:
        df = load_results(args.study)
    else:
        print("Please specify --study, --file, or --compare")
        return
    
    # Analysis
    print_summary_statistics(df)
    print_top_architectures(df, n=args.top_n)
    
    # Plots
    base_dir = Path(__file__).parent.parent / 'experiments'
    plots_dir = base_dir / 'analysis_plots'
    plots_dir.mkdir(exist_ok=True)
    
    save_corr = str(plots_dir / 'correlation_matrix.png') if args.save_plots else None
    save_depth = str(plots_dir / 'depth_width_analysis.png') if args.save_plots else None
    
    plot_correlation_matrix(df, save_path=save_corr)
    plot_depth_vs_width_analysis(df, save_path=save_depth)
    
    # Export LaTeX
    if args.export_latex:
        export_latex_table(df, args.export_latex, n_rows=args.top_n)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
