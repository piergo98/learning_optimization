#!/usr/bin/env python3
"""
Example: Running and Analyzing Architecture Comparison Study

This script demonstrates the complete workflow for comparing neural network
architectures on the switched linear optimization problem.

Run this to see a complete example from start to finish.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_quick_study():
    """Run the quick architecture comparison study."""
    print_section("STEP 1: Running Quick Architecture Study")
    
    print("This will train 7 different neural network architectures and compare them.")
    print("Expected time: 5-15 minutes on GPU, 20-40 minutes on CPU\n")
    
    response = input("Proceed with quick study? (y/n): ")
    if response.lower() != 'y':
        print("Skipping quick study.")
        return False
    
    # Run the quick study
    script_path = Path(__file__).parent / "quick_architecture_comparison.py"
    print(f"\nRunning: python {script_path}\n")
    
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    
    if result.returncode == 0:
        print("\n✓ Quick study completed successfully!")
        return True
    else:
        print("\n✗ Quick study failed!")
        return False


def analyze_quick_results():
    """Analyze the quick study results."""
    print_section("STEP 2: Analyzing Quick Study Results")
    
    results_path = Path(__file__).parent / "quick_study" / "results" / "quick_results.csv"
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(results_path)
    
    print(f"Loaded results for {len(df)} architectures\n")
    
    print("Summary Statistics:")
    print("-" * 70)
    print(f"Best cost difference: {df['mean_cost_diff'].min():.4f}")
    print(f"Mean cost difference: {df['mean_cost_diff'].mean():.4f}")
    print(f"Parameter range: {df['n_params'].min():,} to {df['n_params'].max():,}")
    print(f"Training time range: {df['training_time'].min():.1f}s to {df['training_time'].max():.1f}s")
    
    print("\nTop 3 Architectures by Cost Difference:")
    print("-" * 70)
    top3 = df.nsmallest(3, 'mean_cost_diff')
    for i, row in enumerate(top3.itertuples(), 1):
        print(f"{i}. {row.name}")
        print(f"   Cost difference: {row.mean_cost_diff:.4f}")
        print(f"   Parameters: {row.n_params:,}")
        print(f"   U MSE: {row.mean_u_mse:.6f}")
        print()
    
    print("\nMost Efficient Architecture:")
    print("-" * 70)
    df['efficiency'] = 1.0 / (df['mean_cost_diff'] * np.log10(df['n_params'] + 1))
    best_eff = df.loc[df['efficiency'].idxmax()]
    print(f"{best_eff['name']}")
    print(f"   Cost difference: {best_eff['mean_cost_diff']:.4f}")
    print(f"   Parameters: {best_eff['n_params']:,}")
    print(f"   Efficiency score: {best_eff['efficiency']:.4f}")


def run_comprehensive_study():
    """Run the comprehensive architecture comparison study."""
    print_section("STEP 3: Running Comprehensive Study (Optional)")
    
    print("This will train ~30 architectures with full evaluation.")
    print("Expected time: 2-4 hours on GPU, 8+ hours on CPU\n")
    
    response = input("Proceed with comprehensive study? (y/n): ")
    if response.lower() != 'y':
        print("Skipping comprehensive study.")
        return False
    
    script_path = Path(__file__).parent / "architecture_comparison_study.py"
    print(f"\nRunning: python {script_path}\n")
    
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    
    if result.returncode == 0:
        print("\n✓ Comprehensive study completed successfully!")
        return True
    else:
        print("\n✗ Comprehensive study failed!")
        return False


def demonstrate_custom_analysis():
    """Demonstrate custom analysis of results."""
    print_section("STEP 4: Custom Analysis Example")
    
    results_path = Path(__file__).parent / "quick_study" / "results" / "quick_results.csv"
    
    if not results_path.exists():
        print("No results to analyze. Run the quick study first.")
        return
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    df = pd.read_csv(results_path)
    
    print("Creating custom visualization...")
    
    # Create a custom plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter plot
    ax = axes[0]
    scatter = ax.scatter(
        df['n_params'],
        df['mean_cost_diff'],
        s=df['training_time'] * 10,
        c=df['n_hidden_layers'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
    
    # Annotate best
    best_idx = df['mean_cost_diff'].idxmin()
    best = df.loc[best_idx]
    ax.annotate(
        best['name'],
        xy=(best['n_params'], best['mean_cost_diff']),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Cost Difference (NN - Optimal)', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Architecture Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Hidden Layers')
    
    # Plot 2: Bar chart
    ax = axes[1]
    df_sorted = df.sort_values('mean_cost_diff')
    colors = ['green' if x < 0.05 else 'yellow' if x < 0.10 else 'red' 
              for x in df_sorted['mean_cost_diff']]
    
    bars = ax.bar(range(len(df_sorted)), df_sorted['mean_cost_diff'], color=colors, alpha=0.7)
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(0.05, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='5% suboptimal')
    ax.axhline(0.10, color='yellow', linestyle=':', linewidth=1, alpha=0.5, label='10% suboptimal')
    
    ax.set_xlabel('Architecture (sorted by performance)', fontsize=12)
    ax.set_ylabel('Cost Difference (NN - Optimal)', fontsize=12)
    ax.set_title('Performance Ranking', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['name'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path(__file__).parent / "example_plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / "custom_analysis.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    
    # Export results
    print("\nExporting top 5 architectures to CSV...")
    top5 = df.nsmallest(5, 'mean_cost_diff')
    export_path = plots_dir / "top5_architectures.csv"
    top5.to_csv(export_path, index=False)
    print(f"✓ Saved to {export_path}")


def main():
    """Run the complete example workflow."""
    
    print("\n" + "="*70)
    print("  ARCHITECTURE COMPARISON STUDY - EXAMPLE WORKFLOW")
    print("="*70)
    
    print("\nThis example will guide you through:")
    print("  1. Running a quick architecture comparison study")
    print("  2. Analyzing the results")
    print("  3. (Optional) Running a comprehensive study")
    print("  4. Creating custom visualizations")
    
    # Step 1: Quick study
    quick_success = run_quick_study()
    
    if not quick_success:
        print("\nExample workflow stopped. Run the quick study manually if needed.")
        return
    
    # Step 2: Analyze
    analyze_quick_results()
    
    # Step 3: Comprehensive (optional)
    comprehensive_response = input("\nWould you like to run the comprehensive study? (y/n): ")
    if comprehensive_response.lower() == 'y':
        run_comprehensive_study()
    
    # Step 4: Custom analysis
    custom_response = input("\nWould you like to see custom analysis examples? (y/n): ")
    if custom_response.lower() == 'y':
        demonstrate_custom_analysis()
    
    print_section("WORKFLOW COMPLETE")
    
    print("Next steps:")
    print("  1. Review the generated plots in experiments/quick_study/plots/")
    print("  2. Examine the CSV results in experiments/quick_study/results/")
    print("  3. Modify the architectures in quick_architecture_comparison.py")
    print("  4. Run your own custom analysis using the provided tools")
    print("\nFor more details, see:")
    print("  - experiments/README.md")
    print("  - experiments/ARCHITECTURE_STUDY_GUIDE.md")
    print("  - experiments/QUICK_REFERENCE.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
