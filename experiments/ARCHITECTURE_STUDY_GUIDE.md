# Architecture Comparison Scripts

Two scripts are provided for evaluating neural network architectures on the switched linear optimization problem:

## 1. Quick Study (Recommended for Initial Testing)

**File**: `quick_architecture_comparison.py`

### Features
- Fast execution (~5-15 minutes on GPU)
- Tests 7 representative architectures
- Reduced training epochs (50) and samples (500 train, 20 test)
- Generates single summary plot and CSV results

### Usage
```bash
python experiments/quick_architecture_comparison.py
```

### Architectures Tested
1. **small_1x16**: [1, 16, 11] - Minimal network
2. **medium_2x32**: [1, 32, 32, 11] - Balanced medium
3. **medium_2x64**: [1, 64, 64, 11] - Wider medium
4. **deep_4x32**: [1, 32, 32, 32, 32, 11] - Deep narrow
5. **wide_1x128**: [1, 128, 11] - Shallow wide
6. **pyramid_128_64_32**: [1, 128, 64, 32, 11] - Decreasing width
7. **large_3x128**: [1, 128, 128, 128, 11] - Large capacity

### Output
- `experiments/quick_study/results/quick_results.csv` - Performance metrics
- `experiments/quick_study/plots/quick_comparison.png` - Visual comparison

---

## 2. Comprehensive Study (Production Analysis)

**File**: `architecture_comparison_study.py`

### Features
- Extensive evaluation (~2-4 hours on GPU)
- Tests ~30 architectures systematically
- Full training (200 epochs, 1000 train, 50 test)
- Generates detailed plots, Pareto fronts, and statistics

### Usage
```bash
python experiments/architecture_comparison_study.py
```

### Architecture Families
1. **Varying Depth** (5 architectures)
   - 1-5 hidden layers, constant width=64
   
2. **Varying Width** (5 architectures)
   - 16, 32, 64, 128, 256 neurons, constant depth=3
   
3. **Pyramidal** (4 architectures)
   - Decreasing width patterns
   
4. **Inverted Pyramidal** (4 architectures)
   - Increasing width patterns
   
5. **Small Networks** (3 architectures)
   - Minimal parameter counts
   
6. **Large Networks** (3 architectures)
   - High capacity models

### Output
- `experiments/architecture_study/results/`
  - `architecture_comparison_results.csv` - Main results
  - `architecture_comparison_results.json` - Detailed JSON
  - Individual training histories per architecture
  
- `experiments/architecture_study/plots/`
  - `architecture_comparison.png/pdf` - 6-panel comparison
  - `detailed_comparison.png` - Bar chart comparisons
  - `pareto_front.png` - Efficiency analysis
  
- `experiments/architecture_study/models/`
  - Saved model weights for each architecture

---

## Comparison of Scripts

| Feature | Quick Study | Comprehensive Study |
|---------|-------------|---------------------|
| **Execution time** | 5-15 min | 2-4 hours |
| **Architectures** | 7 | ~30 |
| **Training epochs** | 50 | 200 |
| **Training samples** | 500 | 1000 |
| **Test samples** | 20 | 50 |
| **Plots** | 1 summary | 3 detailed |
| **Use case** | Initial exploration | Publication-quality analysis |

---

## Understanding the Metrics

### Cost Ratio
```
cost_ratio = NN_cost / Optimal_cost
```
- **< 1.05**: Excellent (within 5% of optimal)
- **1.05 - 1.10**: Good (5-10% suboptimality)
- **> 1.10**: Needs improvement

### Control Error (U MSE)
Mean squared error between NN predicted controls and optimal controls
- Lower is better
- Typical good values: < 0.01

### Duration Error (Delta MSE)
Mean squared error between NN predicted mode durations and optimal durations
- Lower is better
- Typical good values: < 0.01

### Efficiency Score
```
efficiency = 1 / (cost_ratio × log10(n_params))
```
Higher values indicate better performance with fewer parameters

---

## Example Workflow

### Step 1: Quick Exploration
```bash
# Run quick study first
python experiments/quick_architecture_comparison.py

# Review results
cat experiments/quick_study/results/quick_results.csv
```

### Step 2: Analyze Quick Results
```python
import pandas as pd
df = pd.read_csv('experiments/quick_study/results/quick_results.csv')
print(df.sort_values('mean_cost_ratio'))
```

### Step 3: Run Comprehensive Study
```bash
# If quick results look promising, run full study
python experiments/architecture_comparison_study.py
```

### Step 4: Deep Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load comprehensive results
df = pd.read_csv('experiments/architecture_study/results/architecture_comparison_results.csv')

# Find best overall
best = df.loc[df['mean_cost_ratio'].idxmin()]
print(f"Best: {best['name']}")
print(f"  Architecture: {best['layer_sizes']}")
print(f"  Cost ratio: {best['mean_cost_ratio']:.4f}")
print(f"  Parameters: {best['n_params']:,}")

# Find most efficient (best performance per parameter)
df['efficiency'] = 1.0 / (df['mean_cost_ratio'] * np.log10(df['n_params']))
efficient = df.loc[df['efficiency'].idxmax()]
print(f"\nMost efficient: {efficient['name']}")
```

---

## Customization

### Modify Problem Setup
Both scripts use similar configuration classes. To change the problem:

```python
# In ExperimentConfig or QuickConfig class
N_PHASES = 20  # Change number of phases
N_CONTROL_INPUTS = 2  # Multi-input problem
X_MIN = -10.0  # Wider state range
X_MAX = 10.0
```

### Add Custom Architectures
In `generate_architectures()` or `generate_quick_architectures()`:

```python
architectures.append({
    'name': 'custom_resnet_like',
    'layer_sizes': [n_inputs, 64, 64, 64, 64, n_outputs],
    'n_hidden_layers': 4,
    'width': 64,
})
```

### Change Training Settings

```python
LEARNING_RATE = 5e-4  # Lower learning rate
N_EPOCHS = 300  # More epochs
BATCH_SIZE = 64  # Larger batches
OPTIMIZER = 'rmsprop'  # Different optimizer
```

---

## Interpreting Plots

### architecture_comparison.png
- **Top row**: Performance metrics vs architecture size
- **Bottom row**: Depth effects and training costs

### pareto_front.png
- Identifies efficient architectures (good performance, low cost)
- Bubble size often indicates secondary metric
- Look for points near bottom-left (low error, low time/params)

### detailed_comparison.png
- Direct bar-chart comparison of all architectures
- Color-coded by performance
- Easy to spot outliers

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16

# Or exclude very large architectures
# Comment out in generate_architectures():
# architectures with > 100k parameters
```

### Slow Optimization (compute_optimal_solution)
```python
# Reduce LBFGS iterations
max_iter=20  # instead of 50 or 100

# Or use cached solutions if X_test is fixed
```

### Poor Network Performance
- Increase training epochs
- Try different learning rates
- Enable data resampling during training
- Check for numerical issues in cost computation

---

## Next Steps

After running these studies, you might want to:

1. **Ablation Studies**: Test effect of activation functions, weight initialization, etc.
2. **Generalization Tests**: Evaluate on out-of-distribution initial states
3. **Constraint Handling**: Add non-negative weight constraints
4. **Multi-objective**: Optimize for both accuracy and model size
5. **Transfer Learning**: Pre-train on simpler problems

---

## Citation

If you use these scripts in your research:

```bibtex
@misc{architecture_comparison_2026,
  title={Architecture Comparison Study for Neural Network Optimal Control},
  author={Pietro [Last Name]},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/username/learning_optimization}
}
```
