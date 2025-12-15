"""
This is an example of using a trained neural network to perform Model Predictive Control (MPC)
over a switched linear system.
"""

import numpy as np
import os, subprocess, sys
import scipy.io
from scipy.linalg import solve_continuous_are
from typing import Optional, Callable, Tuple, Dict, List, Union
import warnings
import json
import matplotlib.pyplot as plt
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU training will not be available.")
    
from src.switched_linear_torch import SwiLin
from src.training import SwiLinNN
from src.switched_system_simulator import SwitchedSystemSimulator



def main():
    start = time.time()
    
    n_phases = 80
    n_control_inputs = 1
    
    dtype = np.float32
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define neural network architecture
    n_NN_inputs = 3  # Example input size
    n_NN_outputs = n_phases * (n_control_inputs + 1)
    model = SwiLinNN(
        layer_sizes=[n_NN_inputs, 50, 50, n_NN_outputs],
        n_phases=n_phases,
    )
    model.to(device)
    
    # Load pre-trained model weights
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'pannocchia_50_50_torch_20251215_173620.pt'))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model to evaluation mode
    model.eval() 
    
    # Create simulation environment
    sim = SwitchedSystemSimulator(
        A=model.sys.A,
        B=model.sys.B,
        dtype=dtype,
    )
    
    # Initial state
    x0 = np.array([[2.0], [-1.0], [5.0]])
    dt = 0.1
    Nsim = 100  # Number of simulation steps
    
    # Simulate closed-loop system
    result = sim.simulate_mpc(x0, model, dt, Nsim)
    
    # Report results
    print(f"Simulation completed with status: {result['message']}")
    print(f"Final state: {result['x'][:, -1]}")
    
    end = time.time()
    print(f"Total execution time: {end - start} seconds")
    
    # Plot results
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    
    sim.plot_trajectory(result, save_path=os.path.join(path, 'nn_mpc_trajectory_pannocchia.png'), show=True)
    sim.plot_phase_portrait(result, save_path=os.path.join(path, 'nn_mpc_phase_pannocchia.png'), show=True)
    print("Plots saved!")
    
    
if __name__ == "__main__":
    main()