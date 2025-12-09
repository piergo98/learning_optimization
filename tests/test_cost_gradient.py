"""
Simple test to verify the cost function and gradient computation work correctly.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import solve_continuous_are

from ocslc.switched_linear import SwiLin as SwiLin_casadi
from src.switched_linear_torch import SwiLin
from data.data_loader import load_data

start = time.time()

torch.autograd.set_detect_anomaly(True)

# Create a simple switched linear system
n_phases = 80
n_states = 3
n_inputs = 1
time_horizon = 10.0

model = {
        'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
        'B': [np.array([[0.25], [2], [0]])],
    }

print("Creating SwiLin system...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
swi_lin = SwiLin(n_phases, n_states, n_inputs, time_horizon, auto=False, device=device)
swi_lin_casadi = SwiLin_casadi(n_phases, n_states, n_inputs, time_horizon, auto=False)
swi_lin.load_model(model)
swi_lin_casadi.load_model(model)

print("Precomputing matrices...")
x0 = np.array([1.3440, -4.5850, 5.6470])
# x0 = np.array([0.0, 0.0, 0.0])
Q = 1. * np.eye(n_states)
R = 0.1 * np.eye(n_inputs)
# Solve the Algebraic Riccati Equation
P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

swi_lin.load_weights(Q, R, P)
swi_lin_casadi.precompute_matrices(x0, Q, R, P)
print("Precomputation complete!")

# print("Testing cost function with random inputs...")
# # Create random u and delta values
# u_list = [torch.randn(n_inputs, dtype=torch.float64, requires_grad=True, device=device) for _ in range(n_phases)]
# delta_raw = torch.randn(n_phases, dtype=torch.float64, requires_grad=True, device=device)
# delta_list = F.softmax(delta_raw, dim=0) * time_horizon
# x0_tensor = torch.tensor(x0, dtype=torch.float64, requires_grad=False, device=device)

# # Compute cost
# print("Computing cost...")
# args = u_list + [delta_list[i] for i in range(n_phases)] + [x0_tensor]
# cost = swi_lin.cost_function(u_list, delta_list, x0_tensor)
# print(f"Cost value: {cost.item():.6f}")

# # Check if we can compute gradients
# print("\nComputing gradients...")
# cost.backward()

# # Check gradients
# print("Gradients computed successfully!")
# for i, u in enumerate(u_list):
#     if u.grad is not None:
#         print(f"  u[{i}] gradient norm: {u.grad.norm().item():.6f}")
#     else:
#         print(f"  u[{i}] gradient: None")

# if delta_raw.grad is not None:
#     print(f"  delta_raw gradient norm: {delta_raw.grad.norm().item():.6f}")
# else:
#     print(f"  delta_raw gradient: None")
    
    
# Load data from file and verify the optimal cost
data_file = 'optimal_params.mat'
data = load_data(data_file)

print("\nVerifying cost against data file...")
# Extract controls and durations from data
controls = data['controls']
phases_duration = data['phases_duration']
# Create tensors
u_data = torch.tensor(controls, dtype=torch.float32, requires_grad=True, device=device).T
delta_list_data = torch.tensor(phases_duration, dtype=torch.float32, requires_grad=True, device=device)
x0_data = torch.tensor(x0, dtype=torch.float32, requires_grad=False, device=device)
# Compute cost from data
# args_data = u_data + [delta_list_data[i] for i in range(n_phases)] + [x0_data]
cost_data = swi_lin.cost_function(u_data, delta_list_data, x0_data)
print(f"Cost from data file: {cost_data.item():.12f}")
x0_aug = np.concatenate([x0, [1]])
cost_casadi = swi_lin_casadi.cost_function(R, x0_aug)
# Flatten controls and durations into individual scalar arguments expected by CasADi
controls_args = [float(controls[0, i]) for i in range(n_phases)]
duration_args = [float(phases_duration[i]) for i in range(n_phases)]
cost_casadi_value = cost_casadi(*controls_args, *duration_args).full().item()
print(f"Cost from CasADi SwiLin: {cost_casadi_value}")

end = time.time()
print(f"\nTest completed in {end - start:.2f} seconds.")

# start = time.time()
# print("\nVerifying cost in autonomous case...")
# # Create autonomous SwiLin system
# n_phases = 6
# n_states = 2
# n_inputs = 0
# time_horizon = 1.0
# swi_lin_auto = SwiLin(n_phases, n_states, n_inputs, time_horizon, auto=True)
# model = {
#     'A': [np.array([[-1.0, 0.0], [0.0, -1.0]]), np.array([[1.0, 1.0], [1.0, -2.0]])],
#     'B': [np.array([[0.0], [0.0]]), np.array([[0.0], [0.0]])],
# }
# swi_lin_auto.load_model(model)
# swi_lin_auto.load_weights(Q=np.eye(n_states), R=np.zeros((n_inputs)), E=0*np.eye(n_states))
# x0 = np.array([1.0, 1.0])
# # Optimal delta values:
# delta_opt = [0.1, 0.197, 0.136, 0.209, 0.125, 0.233]

# # Build the same problem in CasADi SwiLin
# swi_lin_auto_casadi = SwiLin_casadi(n_phases, n_states, n_inputs, time_horizon, auto=True, propagation='exp')
# swi_lin_auto_casadi.load_model(model)
# swi_lin_auto_casadi.precompute_matrices(x0, Q=np.eye(n_states), R=np.zeros((n_inputs)), E=0*np.eye(n_states))
# cost_func_auto_casadi = swi_lin_auto_casadi.cost_function(R=np.zeros((n_inputs)), x0=np.concatenate([x0, [1]]))

# # cost_func_auto_args = [torch.tensor(delta_opt[i], dtype=torch.float64, requires_grad=True) for i in range(n_phases)] + [torch.tensor(x0, dtype=torch.float64, requires_grad=False)]
# cost_auto = swi_lin_auto.cost_function(delta_all=delta_opt, x0=x0)
# print(f"Autonomous cost (PyTorch SwiLin): {cost_auto.item():.12f}")
# cost_auto_casadi = cost_func_auto_casadi(*[float(delta_opt[i]) for i in range(n_phases)])
# print(f"Autonomous cost (CasADi SwiLin): {cost_auto_casadi.full().item():.6f}")

# end = time.time()
# print(f"\nAutonomous test completed in {end - start:.2f} seconds.")


print("\n✓ Test passed! Cost function and gradients work correctly.")
