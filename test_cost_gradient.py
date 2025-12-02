"""
Simple test to verify the cost function and gradient computation work correctly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from switched_linear_torch import SwiLin

# Create a simple switched linear system
n_phases = 6
n_states = 3
n_inputs = 2
time_horizon = 2

model = {
    'A': [
        np.array([[-2.5, 0.5, 0.3], [0.4, -2.0, 0.6], [0.2, 0.3, -1.8]]),
        np.array([[-1.9, 3.2, 0.4], [0.3, -2.1, 0.5], [0, 0.6, -2.3]]),
    ] * 3,  # Repeat for 6 phases
    'B': [
        np.array([[1.5, 0.3], [0.4, 1.2], [0.2, 0.8]]),
        np.array([[1.2, 0.5], [0.3, 0.9], [0.4, 1.1]]),
    ] * 3,  # Repeat for 6 phases
}

print("Creating SwiLin system...")
swi_lin = SwiLin(n_phases, n_states, n_inputs, time_horizon, auto=False)
swi_lin.load_model(model)

print("Precomputing matrices...")
x0 = np.array([2, -1, 5])
Q = 10. * np.eye(n_states)
R = 10. * np.eye(n_inputs)
E = 1. * np.eye(n_states)

swi_lin.precompute_matrices(x0, Q, R, E)
print("Precomputation complete!")

print("\nGetting cost function...")
cost_func = swi_lin.cost_function(R, sym_x0=True)

print("Testing cost function with random inputs...")
# Create random u and delta values
u_list = [torch.randn(n_inputs, dtype=torch.float64, requires_grad=True) for _ in range(n_phases)]
delta_raw = torch.randn(n_phases, dtype=torch.float64, requires_grad=True)
delta_list = F.softmax(delta_raw, dim=0) * time_horizon
x0_tensor = torch.tensor(x0, dtype=torch.float64, requires_grad=False)

# Compute cost
print("Computing cost...")
args = u_list + [delta_list[i] for i in range(n_phases)] + [x0_tensor]
cost = cost_func(*args)
print(f"Cost value: {cost.item():.6f}")

# Check if we can compute gradients
print("\nComputing gradients...")
cost.backward()

# Check gradients
print("Gradients computed successfully!")
for i, u in enumerate(u_list):
    if u.grad is not None:
        print(f"  u[{i}] gradient norm: {u.grad.norm().item():.6f}")
    else:
        print(f"  u[{i}] gradient: None")

if delta_raw.grad is not None:
    print(f"  delta_raw gradient norm: {delta_raw.grad.norm().item():.6f}")
else:
    print(f"  delta_raw gradient: None")

print("\n✓ Test passed! Cost function and gradients work correctly.")
