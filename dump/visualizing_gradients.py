import json
import warnings
import os, subprocess, sys
import time
from typing import Optional, Callable, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.io
from scipy.linalg import solve_continuous_are
from scipy.special import softmax
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU training will not be available.")
    
from ocslc.switched_linear_mpc import SwitchedLinearMPC as SwiLin_casadi

from src.switched_linear_torch import SwiLin
from src.training import SwiLinNN
from src.training import evaluate_cost_functional_batch
from src.training import evaluate_gradient_batch
from matplotlib.patches import Patch

# Global settings
N_PHASES = 10
TIME_HORIZON = 1.0

# NN settings
N_CONTROL_INPUTS = 1
N_STATES = 1
N_NN_INPUTS = 1
N_NN_OUTPUTS = N_PHASES * (N_CONTROL_INPUTS + 1)  # +1 for the mode
EXPERIMENT_DIR = os.path.join(os.path.dirname(__file__), 'visualizing_gradients')
PLOTS_DIR = os.path.join(EXPERIMENT_DIR, 'plots')

# Casadi settings
MULTIPLE_SHOOTING = True
INTEGRATOR = 'exp'
HYBRID = False
PLOT = 'display'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Create necessary directories for the experiment."""
for directory in [EXPERIMENT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load trained model
trained_network = SwiLinNN(
    layer_sizes=[N_NN_INPUTS, 32, 128, 128, N_NN_OUTPUTS],
    n_phases=N_PHASES,
)
trained_network.to(device)
# Load checkpoint
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
checkpoint_path = os.path.abspath(os.path.join(script_dir, '..', 'models', 'analytical_grad_32_32_torch_20260122_164453.pt'))
trained_network.load_state_dict(torch.load(checkpoint_path, map_location=device))


def backpropagation_manual(network, x):
    """
    Implement backpropagation algorithm to compute gradients using analytical cost functional.
    
    Args:
        network: SwiLinNN neural network model
        x: Input tensor (batch_size, input_dim) - initial states x0
    
    Returns:
        Dictionary containing gradients for each layer's weights and biases
    """
    # Get all linear layers from the network MLP
    layers = [module for module in network.layers]
    L = len(layers)  # Number of layers
    
    # Storage for forward pass
    a = {}  # Activations
    z = {}  # Pre-activation values
    
    # ========== FORWARD PASS ==========
    input = x.detach().clone()  # Input
    output, acts = network(input)  # Forward pass through the network
    
    for a in acts:
        a.retain_grad()  # Ensure gradients are retained for each layer
    
    # Compute loss using analytical cost functional (LQR-style)
    n_phases = network.n_phases
    n_inputs = network.sys.n_inputs
    n_control_outputs = n_phases * n_inputs
    
    # Extract controls and deltas
    controls = output[:, :n_control_outputs]
    delta_raw = output[:, n_control_outputs:]
    
    # Process deltas (same as in training)
    last = delta_raw[:, -1:]
    delta_raw_traslated = delta_raw - last
    delta_normalized = F.softmax(delta_raw_traslated, dim=-1)
    T_tensor = torch.tensor(network.sys.time_horizon, device=x.device, dtype=x.dtype)
    deltas = delta_normalized * T_tensor
    
    # Soft-clip controls (same as in training)
    u_min = -1.0
    u_max = 1.0
    u_center = (u_max + u_min) / 2.0
    u_range = (u_max - u_min) / 2.0
    controls = u_center + u_range * torch.tanh(controls)
    
    # Reshape for cost computation
    B_batch = x.shape[0]
    controls_reshaped = controls.view(B_batch, n_phases, n_inputs)
    deltas_batch = deltas.view(B_batch, n_phases)
    
    # Compute LQR cost functional using the analytical approach
    J_batch = evaluate_cost_functional_batch(network.sys, controls_reshaped, deltas_batch, x)
    loss = J_batch.mean()
    
    # ========== BACKWARD PASS ==========
    # For analytical gradient, we use PyTorch's autograd on the cost functional
    # This computes gradients through the entire computation graph and the activations gradients 
    grads = []
    gradients = {}
    activations = {}
    
    # Compute gradients using autograd
    loss.backward()
    
    # Extract gradients from each layer
    for l in range(L, 0, -1):
        layer = layers[l-1]
        if layer.weight.grad is not None:
            gradients[f'W_{l}'] = layer.weight.grad.clone()
        if layer.bias is not None and layer.bias.grad is not None:
            gradients[f'b_{l}'] = layer.bias.grad.clone()
    
    # Unpack activations (loss gradients w.r.t. activations)
    for i, a in enumerate(acts):
        activations[f'a_{i+1}'] = a.grad.clone()
        
    # Compute the gradient wrt the network output using analytical cost functional
    grad_u, grad_delta = evaluate_gradient_batch(trained_network.sys, controls_reshaped, deltas_batch, x)
    grad_output = torch.cat([grad_u.view(B_batch, -1), grad_delta.view(B_batch, -1)], dim=-1)
    grad_output_pinv = torch.linalg.pinv(grad_output)
    for p in network.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    full_grad = torch.cat(grads)

    output_jacobian = torch.mul(grad_output_pinv, full_grad.unsqueeze(0))
    
    return gradients, activations, output_jacobian, L


# Use the trained network for gradient computation
x0_tensor = torch.tensor([[2.0]], dtype=trained_network.sys.dtype, device=device)

# Compute gradients using manual backpropagation with analytical cost functional
gradients, activations, output_jacobian, n_layers = backpropagation_manual(trained_network, x0_tensor)


# ============================================================================
# COMPACT GRADIENT VISUALIZATION: Weight Heatmaps + Activation Gradients
# ============================================================================

import matplotlib.gridspec as gridspec

# Create figure with 2 columns: weight gradients (left) and activation gradients (right)
num_layers = n_layers
fig = plt.figure(figsize=(14, max(4, 2 * num_layers)))
gs = gridspec.GridSpec(num_layers, 1, width_ratios=[1], wspace=0.4, hspace=0.6)

# First figure: Weight gradient heatmaps per layer
for layer_idx in range(1, num_layers + 1):
    ax = fig.add_subplot(gs[layer_idx - 1, 0])
    W_grad = gradients[f'W_{layer_idx}'].cpu().numpy()
    print(W_grad.shape)
    
    # Symmetric colormap centered at zero
    vmax = np.max(np.abs(W_grad)) if W_grad.size > 0 else 1.0
    im = ax.imshow(W_grad, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f'Layer {layer_idx} — ∂L/∂W (shape {W_grad.shape})', fontsize=10)
    ax.set_xlabel('Input features')
    ax.set_ylabel('Output neurons')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

plt.savefig(os.path.join(PLOTS_DIR, 'architecture_comparison.png'), dpi=300, bbox_inches='tight')
    
    
# Third figure: Output Jacobian wrt network weights
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
output_jacobian_np = output_jacobian.detach().cpu().numpy()
tol = 1e-12
mask = np.abs(output_jacobian_np) > tol
rows, cols = np.nonzero(mask)

# Map flattened parameter indices to layer groups
param_shapes = [tuple(p.shape) for p in trained_network.parameters()]
param_sizes = [int(np.prod(s)) for s in param_shapes]
cumsum = np.cumsum(param_sizes)
# For each column index, find which parameter (layer group) it belongs to
col_group_ids = np.searchsorted(cumsum, cols, side='right')  # 0-based group id

# Colormap for groups
n_groups = len(param_sizes)
cmap = plt.cm.get_cmap('tab20', max(1, n_groups))

# Draw background vertical bands for each parameter group (lightly filled)
start_idx = 0
legend_handles = []
for i, size in enumerate(param_sizes):
    end_idx = start_idx + size
    color = cmap(i % cmap.N)
    ax.axvspan(start_idx - 0.5, end_idx - 0.5, facecolor=color, alpha=0.12, zorder=0)
    legend_handles.append(Patch(facecolor=color, edgecolor='k', label=f'Param {i+1}: {param_shapes[i]}'))
    start_idx = end_idx

# Scatter black dots for nonzero entries
ax.scatter(cols, rows, s=6, marker='.', c='k', linewidths=0, zorder=5)

# Axis formatting
ax.set_yticks(np.arange(0, output_jacobian_np.shape[0]))
ax.set_xlim(-0.5, output_jacobian_np.shape[1] - 0.5)
ax.invert_yaxis()
ax.set_xlabel('Weight index (flattened parameters)')
ax.set_ylabel('Output dimension')
ax.set_title(f'Nonzero pattern of output_jacobian (shape {output_jacobian_np.shape})')
ax.set_aspect('auto')

# Legend: one entry per parameter (layer) group
ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8)
plt.savefig(os.path.join(PLOTS_DIR, 'output_jacobian_structure.png'), dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
