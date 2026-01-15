# Architecture Comparison Study

This directory contains experiments for evaluating how neural network architecture affects performance on the switched linear optimization problem.

## Overview

The `architecture_comparison_study.py` script trains multiple neural networks (SwiLinNN) with varying architectures and compares their performance against the optimal solution computed via gradient-based optimization.

### What is Evaluated

The script systematically varies:
- **Number of hidden layers** (depth)
- **Neurons per layer** (width)
- **Total number of parameters**
- **Architecture patterns** (uniform, pyramidal, inverted pyramidal)

For each architecture, the script measures:
- **Cost function value** (compared to optimal)
- **Control input errors** (u)
- **Mode duration errors** (delta)
- **Training time**
- **Convergence behavior**

## Architecture Families Tested

1. **Varying Depth** (constant width = 64)
   - 1 to 5 hidden layers

2. **Varying Width** (constant depth = 3)
   - 16, 32, 64, 128, 256 neurons per layer

3. **Pyramidal** (decreasing width)
   - Example: [256, 128, 64, 32]

4. **Inverted Pyramidal** (increasing width)
   - Example: [32, 64, 128, 256]

5. **Small Networks**
   - Minimal architectures with few parameters

6. **Large Networks**
   - High-capacity architectures

## Usage

### Basic Usage

```bash
cd /home/pietro/data-driven/learning_optimization
python experiments/architecture_comparison_study.py
```

### Configuration

Key parameters can be modified in the `ExperimentConfig` class:

```python
class ExperimentConfig:
    # Problem setup
    N_PHASES = 10
    N_CONTROL_INPUTS = 1
    
    # Training setup
    N_SAMPLES_TRAIN = 1000
    N_SAMPLES_VAL = 100
    N_SAMPLES_TEST = 50
    LEARNING_RATE = 1e-3
    N_EPOCHS = 200
    BATCH_SIZE = N_SAMPLES_TRAIN
    
    # Data range
    X_MIN = -5.0
    X_MAX = 5.0
```

## Output Structure

The script creates the following directory structure:

```
experiments/architecture_study/
├── results/
│   ├── architecture_comparison_results.csv
│   ├── architecture_comparison_results.json
│   └── [architecture_name]_[timestamp]/
│       └── history.json
├── models/
│   └── [architecture_name]_[timestamp].pt
└── plots/
    ├── architecture_comparison.png
    ├── architecture_comparison.pdf
    ├── detailed_comparison.png
    └── pareto_front.png
```

## Results Files

### CSV/JSON Results

The main results file contains:
- `name`: Architecture identifier
- `layer_sizes`: Network topology
- `n_hidden_layers`: Number of hidden layers
- `n_params`: Total trainable parameters
- `training_time`: Wall-clock training time (seconds)
- `final_train_loss`: Final training loss
- `final_val_loss`: Final validation loss
- `mean_cost_nn`: Average cost from NN predictions
- `mean_cost_opt`: Average optimal cost
- `mean_cost_ratio`: Ratio NN/Optimal (1.0 = perfect)
- `mean_u_mse`: Mean squared error in control inputs
- `mean_delta_mse`: Mean squared error in mode durations

### Plots

1. **architecture_comparison.png**
   - 6-panel comparison showing:
     - Cost vs number of parameters
     - Cost ratio vs number of parameters
     - Control error vs number of parameters
     - Duration error vs number of parameters
     - Cost vs network depth
     - Training time vs number of parameters

2. **detailed_comparison.png**
   - Bar charts comparing all architectures
   - Cost, error, and performance scores

3. **pareto_front.png**
   - Trade-off between performance and computational cost
   - Helps identify efficient architectures

## Interpreting Results

### Key Metrics

- **Cost Ratio < 1.1**: Network achieves near-optimal performance
- **U MSE < 0.01**: Good control approximation
- **Delta MSE < 0.01**: Good mode duration approximation

### Efficiency Analysis

The script computes an efficiency score:
```
efficiency = 1.0 / (cost_ratio * log10(n_params))
```

Higher efficiency means better performance with fewer parameters.

## Example Analysis

After running the experiment, you can analyze results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('experiments/architecture_study/results/architecture_comparison_results.csv')

# Find best architectures
best_cost = df.loc[df['mean_cost_ratio'].idxmin()]
print(f"Best architecture: {best_cost['name']}")
print(f"Cost ratio: {best_cost['mean_cost_ratio']:.4f}")

# Plot custom analysis
plt.figure(figsize=(10, 6))
plt.scatter(df['n_params'], df['mean_cost_ratio'], 
            s=100, alpha=0.6, c=df['n_hidden_layers'], cmap='viridis')
plt.xscale('log')
plt.xlabel('Number of Parameters')
plt.ylabel('Cost Ratio (NN/Optimal)')
plt.colorbar(label='Hidden Layers')
plt.title('Architecture Performance')
plt.show()
```

## Extending the Study

To add custom architectures:

```python
# In generate_architectures() function
architectures.append({
    'name': 'custom_arch',
    'layer_sizes': [n_inputs, 100, 200, 100, n_outputs],
    'n_hidden_layers': 3,
    'width': 'custom',
})
```

To test different activation functions:

```python
# In train_architecture() function
network = SwiLinNN(
    layer_sizes=arch['layer_sizes'],
    n_phases=config.N_PHASES,
    activation='tanh',  # or 'sigmoid'
    output_activation='linear'
)
```

## Performance Tips

1. **GPU Acceleration**: The script automatically uses CUDA if available
2. **Parallel Runs**: For faster execution, split architectures across multiple jobs
3. **Early Stopping**: Enabled by default to save time
4. **Checkpointing**: Models and histories are saved automatically

## Troubleshooting

### Out of Memory

Reduce batch size or number of large architectures:
```python
BATCH_SIZE = 16  # instead of 32
```

### Slow Training

Enable GPU or reduce epochs:
```python
N_EPOCHS = 100  # instead of 200
```

### Poor Convergence

Try different learning rates or optimizers:
```python
LEARNING_RATE = 1e-4
OPTIMIZER = 'adam'  # or 'sgd', 'rmsprop'
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{architecture_comparison_study,
  title={Neural Network Architecture Comparison for Switched Linear Optimization},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/learning_optimization}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
