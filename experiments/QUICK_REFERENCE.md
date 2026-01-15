# Neural Network Architecture Study - Complete Guide

This directory contains a comprehensive framework for evaluating how neural network architecture affects performance on switched linear optimization problems.

## 📁 Files Overview

### Core Scripts

1. **`quick_architecture_comparison.py`** ⚡
   - Fast exploration (5-15 minutes)
   - 7 representative architectures
   - Good for initial testing

2. **`architecture_comparison_study.py`** 🔬
   - Comprehensive analysis (2-4 hours)
   - ~30 architectures across 6 families
   - Publication-quality results

3. **`analyze_results.py`** 📊
   - Post-processing and visualization
   - Compare multiple runs
   - Export tables and plots

### Documentation

- **`README.md`** - Detailed usage instructions
- **`ARCHITECTURE_STUDY_GUIDE.md`** - Comprehensive guide with examples
- **This file** - Quick reference

---

## 🚀 Quick Start

### Option 1: Fast Test (Recommended First)

```bash
cd /home/pietro/data-driven/learning_optimization
python experiments/quick_architecture_comparison.py
```

**What it does:**
- Trains 7 different neural network architectures
- Compares them against optimal solutions
- Generates summary plots and CSV results
- Takes ~10 minutes on GPU

**Output:**
```
experiments/quick_study/
├── results/quick_results.csv
└── plots/quick_comparison.png
```

### Option 2: Comprehensive Analysis

```bash
python experiments/architecture_comparison_study.py
```

**What it does:**
- Systematically tests ~30 architectures
- Varies depth, width, and patterns
- Generates detailed statistical analysis
- Takes ~3 hours on GPU

**Output:**
```
experiments/architecture_study/
├── results/
│   ├── architecture_comparison_results.csv
│   └── architecture_comparison_results.json
├── models/
│   └── [saved model checkpoints]
└── plots/
    ├── architecture_comparison.png
    ├── detailed_comparison.png
    └── pareto_front.png
```

### Option 3: Analyze Existing Results

```bash
# Analyze quick study
python experiments/analyze_results.py --study quick --save-plots

# Analyze comprehensive study
python experiments/analyze_results.py --study comprehensive --save-plots --export-latex table.tex

# Compare two runs
python experiments/analyze_results.py --compare run1.csv run2.csv
```

---

## 📊 What Gets Measured

### For Each Architecture

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Cost Ratio** | NN cost / Optimal cost | < 1.05 |
| **U MSE** | Control input error | < 0.01 |
| **Delta MSE** | Mode duration error | < 0.01 |
| **Training Time** | Wall-clock time | Lower better |
| **Parameters** | Total trainable params | Depends on use case |

### Comparison Metrics

- **Efficiency**: Performance per parameter
- **Pareto optimality**: Best trade-offs
- **Depth vs Width**: Systematic analysis
- **Convergence**: Training dynamics

---

## 🎯 Key Research Questions Answered

1. **Does depth or width matter more?**
   - Compare architectures with same parameters but different depth/width ratios

2. **What's the minimum viable architecture?**
   - Find smallest network achieving < 5% suboptimality

3. **Diminishing returns?**
   - Identify parameter count where performance plateaus

4. **Shape matters?**
   - Pyramidal vs uniform vs inverted pyramidal

5. **Training efficiency?**
   - Best accuracy/time trade-off

---

## 📈 Example Results Interpretation

After running the quick study, you might see:

```
BEST ARCHITECTURE: medium_2x64
Cost ratio: 1.023
Parameters: 4,416
Training time: 12.3s
U MSE: 0.003214
```

**Interpretation:**
- ✅ Within 2.3% of optimal (excellent!)
- ✅ Only 4.4k parameters (very efficient)
- ✅ Fast training (< 15s)
- ✅ Low control error

---

## 🔧 Common Customizations

### Change Problem Setup

```python
# In QuickConfig or ExperimentConfig class
N_PHASES = 20  # More time phases
N_CONTROL_INPUTS = 2  # Multi-input problem
X_MIN, X_MAX = -10.0, 10.0  # Wider state space
```

### Add Custom Architecture

```python
# In generate_quick_architectures() or generate_architectures()
architectures.append({
    'name': 'my_custom_arch',
    'layer_sizes': [1, 128, 64, 32, 11],
    'n_hidden_layers': 3,
})
```

### Modify Training

```python
N_EPOCHS = 100  # More/fewer epochs
LEARNING_RATE = 5e-4  # Different learning rate
BATCH_SIZE = 64  # Larger batches
OPTIMIZER = 'rmsprop'  # Try different optimizer
```

---

## 📋 Typical Workflow

### 1️⃣ Initial Exploration (Day 1)

```bash
# Run quick study
python experiments/quick_architecture_comparison.py

# Review results
python experiments/analyze_results.py --study quick

# Identify promising architectures
```

### 2️⃣ Deep Dive (Day 2)

```bash
# Run comprehensive study
python experiments/architecture_comparison_study.py

# Generate detailed analysis
python experiments/analyze_results.py --study comprehensive --save-plots
```

### 3️⃣ Refinement (Day 3)

```python
# Based on results, test variations
# Modify generate_architectures() to focus on best patterns
# Re-run with more epochs or samples
```

### 4️⃣ Publication

```bash
# Export best results
python experiments/analyze_results.py --study comprehensive \
    --export-latex results_table.tex \
    --save-plots

# Copy plots from experiments/architecture_study/plots/
```

---

## 🎓 Understanding the Output

### CSV Results

Each row contains:
```
name,layer_sizes,n_hidden_layers,n_params,training_time,
final_train_loss,final_val_loss,mean_cost_nn,mean_cost_opt,
mean_cost_ratio,mean_u_mse,mean_delta_mse
```

**Key columns:**
- `mean_cost_ratio`: Primary performance metric (lower = better)
- `mean_u_mse`, `mean_delta_mse`: Solution quality
- `n_params`: Model size
- `training_time`: Computational cost

### Plots Explained

**architecture_comparison.png**
- 6 subplots showing different perspectives
- Look for: sweet spots, diminishing returns, outliers

**pareto_front.png**
- Trade-off between performance and cost
- Points near bottom-left are most efficient

**detailed_comparison.png**
- Direct bar chart comparison
- Color-coded by performance tiers

---

## ⚠️ Troubleshooting

### Problem: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 16

# Or skip very large architectures
# Comment out networks with > 100k parameters
```

### Problem: Poor Network Performance

**Solutions:**
1. Increase training epochs
2. Try different learning rates (1e-4, 5e-4, 1e-3)
3. Enable early stopping
4. Check data distribution

### Problem: Slow Execution

**Solutions:**
1. Use GPU if available
2. Reduce N_SAMPLES_TRAIN
3. Use quick study instead of comprehensive
4. Parallelize across multiple GPUs/machines

---

## 📚 Advanced Usage

### Custom Metrics

Add your own evaluation metrics:

```python
# In evaluate_network_on_test_set()
results['custom_metric'] = []

for x0 in X_test:
    # Your custom evaluation
    custom_value = compute_custom_metric(network, x0)
    results['custom_metric'].append(custom_value)
```

### Different Optimization Baselines

Replace the optimal solution computation:

```python
def compute_optimal_solution(x0, config):
    # Your custom optimization method
    # Could use: CasADi, IPOPT, other solvers
    ...
    return {'u': u_opt, 'delta': delta_opt, 'cost': cost_opt}
```

### Multi-objective Optimization

Modify to optimize for multiple objectives:

```python
# Add Pareto dominance analysis
df['pareto_rank'] = compute_pareto_rank(
    df[['mean_cost_ratio', 'n_params', 'training_time']]
)
```

---

## 📖 Example Analysis Script

After running the study:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('experiments/architecture_study/results/architecture_comparison_results.csv')

# Find best by cost
best = df.loc[df['mean_cost_ratio'].idxmin()]
print(f"Best: {best['name']}")
print(f"  Cost ratio: {best['mean_cost_ratio']:.4f}")
print(f"  Parameters: {best['n_params']:,}")

# Efficiency score
df['efficiency'] = 1.0 / (df['mean_cost_ratio'] * np.log10(df['n_params']))
efficient = df.loc[df['efficiency'].idxmax()]
print(f"\nMost efficient: {efficient['name']}")

# Plot custom comparison
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df['n_params'],
    df['mean_cost_ratio'],
    s=200,
    c=df['training_time'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='black'
)
ax.axhline(1.0, color='red', linestyle='--', label='Optimal')
ax.set_xlabel('Number of Parameters')
ax.set_ylabel('Cost Ratio (NN/Optimal)')
ax.set_xscale('log')
ax.set_title('Architecture Performance Landscape')
plt.colorbar(scatter, label='Training Time (s)')
plt.legend()
plt.tight_layout()
plt.show()

# Export top 10 for paper
top10 = df.nsmallest(10, 'mean_cost_ratio')
top10.to_latex('top10_architectures.tex', index=False)
```

---

## 🤝 Contributing

To add new architecture families:

1. Modify `generate_architectures()` in the main script
2. Add documentation to this guide
3. Update expected ranges in analysis tools
4. Test on small dataset first

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example outputs in documentation
3. Open an issue on GitHub
4. Contact maintainers

---

## ✅ Checklist for Complete Study

- [ ] Run quick study to validate setup
- [ ] Review quick results for sanity check
- [ ] Run comprehensive study overnight
- [ ] Analyze results with analyze_results.py
- [ ] Generate plots for presentation
- [ ] Export LaTeX tables
- [ ] Document best architectures
- [ ] Save trained models
- [ ] Archive results with timestamp

---

## 🎉 Expected Outcomes

After completing this study, you should have:

1. **Quantitative understanding** of architecture effects
2. **Best architecture** for your specific problem
3. **Trade-off curves** (Pareto fronts)
4. **Efficiency metrics** for model selection
5. **Publication-ready** figures and tables
6. **Trained models** ready for deployment

---

**Last Updated:** January 2026  
**Maintainer:** Pietro  
**License:** See repository LICENSE
