import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import torch

from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwiLin
from optimizers import SGD, Adam, RMSProp
from optimizers import gpu_optimize

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")


def switched_problem(n_phases=5):
    """
    Set up a switched linear problem and compute cost and gradient functions
    """
    model = {
        'A': [
            np.array([[-2.5, 0.5, 0.3], [0.4, -2.0, 0.6], [0.2, 0.3, -1.8]]),
            np.array([[-1.9, 3.2, 0.4], [0.3, -2.1, 0.5], [0, 0.6, -2.3]]),
            np.array([[-2.2, 0, 0.5],   [0.2, -1.7, 0.4], [0.3, 0.2, -2.0]]),
            np.array([[-1.8, 0.3, 0.2], [0.5, -2.4, 0],   [0.4, 0, -2.2]]),
            np.array([[-2.0, 0.4, 0],   [0.3, -2.2, 0.2], [0.5, 0.3, -1.9]]),
            np.array([[-2.3, 0.2, 0.3], [0, -2.0, 0.4],   [0.2, 0.5, -2.1]]),
            np.array([[-1.7, 0.5, 0.4], [0.2, -2.5, 0.3], [1.1, 0.2, -2.4]]),
            np.array([[-2.1, 0.3, 0.2], [0.4, -1.9, 0.5], [0.3, 0.1, -2.0]]),
            np.array([[-2.4, 0, 0.5],   [0.2, -2.3, 0.3], [0.4, 0.2, -1.8]]),
            np.array([[-1.8, 0.4, 0.3], [0.5, -2.1, 0.2], [0.2, 3.1, -2.2]]),
        ],
        'B': [
            np.array([[1.5, 0.3], [0.4, 1.2], [0.2, 0.8]]),
            np.array([[1.2, 0.5], [0.3, 0.9], [0.4, 1.1]]),
            np.array([[1.0, 0.4], [0.5, 1.3], [0.3, 0.7]]),
            np.array([[1.4, 0.2], [0.6, 1.0], [0.1, 0.9]]),
            np.array([[1.3, 0.1], [0.2, 1.4], [0.5, 0.6]]),
            np.array([[1.1, 0.3], [0.4, 1.5], [0.2, 0.8]]),
            np.array([[1.6, 0.2], [0.3, 1.1], [0.4, 0.7]]),
            np.array([[1.0, 0.4], [0.5, 1.2], [0.3, 0.9]]),
            np.array([[1.2, 0.5], [0.1, 1.3], [0.6, 0.8]]),
            np.array([[1.4, 0.3], [0.2, 1.0], [0.5, 0.7]]),
        ],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 10

    x0 = np.array([2, -1, 5])
    
    xr = np.array([1, -3])
    
    swi_lin = SwiLin(
        n_phases, 
        n_states,
        n_inputs,
        time_horizon, 
        auto=False, 
    )
    
    # Load model
    swi_lin.load_model(model)

    Q = 10. * np.eye(n_states)
    R = 10. * np.eye(n_inputs)
    E = 1. * np.eye(n_states)

    swi_lin.precompute_matrices(x0, Q, R, E)
    x0 = np.append(x0, 1)  # augment with 1 for affine term
    J_func = swi_lin.cost_function(R, x0)
        
    grad_J_u = []
    grad_J_delta = []

    for k in range(n_phases):
        # Compute gradient of the cost
        du, d_delta = swi_lin.grad_cost_function(k, R)
        # print(f"Length du: {len(du)}")

        grad_J_u += du
        grad_J_delta.append(d_delta)

    grad_J = ca.vertcat(*grad_J_delta, *grad_J_u)

    # keep the original stacked forms if needed
    grad_J_u = np.hstack(grad_J_u)
    grad_J_delta = np.hstack(grad_J_delta)
    
    # Create a CasADi function for the gradient
    grad_J_func = ca.Function('grad_J', [*swi_lin.u, *swi_lin.delta], [grad_J])
    
    # Create wrapper functions for the optimizer
    def cost_function(params, indices=None, data=None):
        """
        params: 1D array (n_params,) or 2D array (batch_size, n_params)
        indices: optional indices to select a minibatch from a 2D params array
        Returns the scalar loss (averaged over minibatch if batch provided).
        """
        # Wrapper that convert torch -> numpy -> CasADi -> numpy -> torch
        params_np = params.detach().cpu().numpy()
        # From a single flattened params vector, unpack into controls and durations
        u = params_np[:n_phases * n_inputs].reshape((n_phases, n_inputs)).tolist()
        phases_duration = params_np[n_phases * n_inputs:].reshape((n_phases,)).tolist()
        params_list = u + phases_duration
        # Compute the cost function for a single example (no batch)
        J_np = float(J_func(*params_list).full().item())
        
        if data is not None:
            u = data['controls'].ravel()
            phases_duration = data['phases_duration'].ravel()
            params_ref = np.concatenate([u, phases_duration])
            # print(f"Reference params: {params_ref}")
            
            # Compute the loss wrt reference params
            params_ref = np.asarray(params_ref)
            # add numpy sum of squared differences to scalar loss
            J_np += float(np.sum((params_np - params_ref) ** 2) / len(params_ref))
            
        # Convert back to torch tensor on same device as input
        J = torch.tensor(J_np, dtype=torch.float32, device=params.device)

        return J

    def gradient_function(params, indices=None, data=None):
        """
        params: 1D array (n_params,) or 2D array (batch_size, n_params)
        indices: optional indices to select a minibatch from a 2D params array
        Returns gradient vector (n_params,) averaged over minibatch if batch provided.
        """
        # Wrapper that convert torch -> numpy -> CasADi -> numpy -> torch
        params_np = params.detach().cpu().numpy()
        # From a single flattened params vector, unpack into controls and durations
        u = params_np[:n_phases * n_inputs].reshape((n_phases, n_inputs)).tolist()
        phases_duration = params_np[n_phases * n_inputs:].reshape((n_phases,)).tolist()
        params_list = u + phases_duration
        # single example
        grad_J_np = np.asarray(grad_J_func(*params_list).full().ravel())
        
        if data is not None:
            u = data['controls'].ravel()
            phases_duration = data['phases_duration'].ravel()
            params_ref = np.concatenate([u, phases_duration])
            params_ref = np.asarray(params_ref)
            # add gradient of the squared differences to grad_J
            grad_J_np += 2 * (params_np - params_ref)
            
        # Convert back to torch tensor on same device as input
        grad_J = torch.tensor(grad_J_np, dtype=torch.float32, device=params.device)
            
        return grad_J

    return J_func, grad_J_func, cost_function, gradient_function

def params_optimization(optimizer="sgd", data=None):
    """
    Perform optimization using the switched linear problem setup.
    """
    n_phases = 50

    _, _, cost_function, gradient_function = switched_problem(n_phases)
    
    # If data is provided, build the initial parameters by perturbating the reference
    if data is not None:
        u = data['controls'].ravel()
        phases_duration = data['phases_duration'].ravel()
        true_params = np.concatenate([u, phases_duration])
        # Add some noise to the initial parameters
        initial_params = np.random.normal(true_params, 0.1, true_params.shape)
    else:
        initial_params = np.ones(n_phases * (2 + 1))  # 2 inputs + 1 duration per phase
        
    # Evaluate initial cost and gradient
    # initial_cost = cost_function(torch.tensor(initial_params, dtype=torch.float32, device=device), data=data).item()
    # initial_grad = gradient_function(torch.tensor(initial_params, dtype=torch.float32, device=device), data=data).cpu().numpy()
    # print("Initial parameters:", initial_params)
    # print(f"Initial cost: {initial_cost}")
    # print(f"Initial gradient norm: {np.linalg.norm(initial_grad)}")
    # input("Press Enter to start optimization...")

    # Choose learning rate schedule
    schedules = [
        {'name': 'Constant', 'schedule': 'constant'},
        {'name': 'Step', 'schedule': 'step', 'params': {'step_size': 20, 'gamma': 0.5}},
        {'name': 'Exponential', 'schedule': 'exponential', 'params': {'decay_rate': 0.05}},
        {'name': 'Inverse', 'schedule': 'inverse', 'params': {'decay_rate': 0.1}},
    ]
    schedule = schedules[0]['schedule']
    params = schedules[0].get('params', {})
    
    optimizers = [
        ('SGD', {'momentum': 0.9, 'nesterov': True}),
        ('Adam', {'beta1': 0.9, 'beta2': 0.999}),
        ('RMSProp', {'rho': 0.9}),
    ]
    
    # Gather optimizer-specific parameters based on the chosen optimizer
    optimizer_params = optimizers[0][1] if optimizer == "sgd" else optimizers[1][1] if optimizer == "adam" else optimizers[2][1]

    params_optimized, history = gpu_optimize(
            params_init=initial_params,
            gradient_func=gradient_function,
            optimizer=optimizer,
            loss_func=cost_function,  # optional
            learning_rate=5e-5,
            n_epochs=int(1e5),
            data=data,
            device=device,
            verbose=True,
            **optimizer_params,
    )
    
    # Convert optimized params to numpy for plotting/printing
    if torch.is_tensor(params_optimized):
        params_optimized = params_optimized.detach().cpu().numpy()

    # Print results
    # print("Optimized Parameters:", params_optimized)
    # Get n_inputs from data or infer from params shape
    n_inputs = data['n_inputs'] if data is not None else 2
    plot_params(params_optimized, n_phases, n_inputs)
    plot_history(history)
    
def plot_history(history):
    """
    Plot the evolution of each component of the optimization history.
    history = {loss, learning_rate, gradient_norm}
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Loss
    axes[0].plot(history['loss'], color='tab:red', lw=2, label='Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()

    # Gradient norm
    axes[1].plot(history['gradient_norm'], color='tab:blue', lw=2, label='Gradient Norm')
    axes[1].set_ylabel('Gradient Norm')
    axes[1].grid(True)
    axes[1].legend()

    # Learning rate
    axes[2].plot(history['learning_rate'], color='tab:green', lw=2, label='Learning Rate')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True)
    axes[2].legend()

    plt.suptitle('Optimization History', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_params(params_optimized, n_phases, n_inputs=1):
    """
    Plot the optimized parameters (inputs and durations) over phases.
    
    Args:
        params_optimized: flattened array [u_0_0, u_0_1, ..., u_{n-1}_0, u_{n-1}_1, delta_0, ..., delta_{n-1}]
        n_phases: number of phases
        n_inputs: number of control inputs per phase
    """
    # Split into controls and durations
    n_control_params = n_phases * n_inputs
    controls = params_optimized[:n_control_params].reshape((n_phases, n_inputs))
    durations = params_optimized[n_control_params:]

    # Create subplots: n_inputs plots for controls + 1 for durations
    fig, axes = plt.subplots(n_inputs + 1, 1, figsize=(10, 3 * (n_inputs + 1)), sharex=True)
    
    # If only one subplot, axes is not an array
    if n_inputs == 0:
        axes = [axes]

    # Plot each control input
    colors = ['tab:orange', 'tab:cyan', 'tab:green', 'tab:red']
    for i in range(n_inputs):
        axes[i].step(range(n_phases), controls[:, i], where='post', 
                     color=colors[i % len(colors)], lw=2)
        axes[i].set_ylabel(f'Control Input {i+1}')
        axes[i].set_title(f'Optimized Control Input {i+1} over Phases')
        axes[i].grid(True)

    # Plot durations
    axes[n_inputs].step(range(n_phases), durations, where='post', color='tab:purple', lw=2)
    axes[n_inputs].set_ylabel('Phase Duration')
    axes[n_inputs].set_xlabel('Phase')
    axes[n_inputs].set_title('Optimized Phase Durations over Phases')
    axes[n_inputs].grid(True)

    plt.tight_layout()
    plt.show()
    
def load_data(filename):
    """
    Load data from file <filename>, which has to be in the data folder.
    The function loads both csv or mat files
    """
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(data_folder, filename)

    if filename.endswith('.csv'):
        loaded_data = np.loadtxt(file_path, delimiter=',')
    elif filename.endswith('.mat'):
        loaded_data = scipy.io.loadmat(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .mat files.")
    
    # Handle data after loading
    keys_to_keep = ['n_phases', 'controls', 'phases_duration']

    data = {k: loaded_data[k] for k in keys_to_keep}
    # Normalize and reshape controls into shape (n_inputs, n_phases)
    controls = np.asarray(data['controls']).ravel()
    n_phases = int(np.squeeze(np.asarray(data['n_phases'])))
    if controls.size % n_phases != 0:
        raise ValueError(f"Controls length ({controls.size}) is not divisible by n_phases ({n_phases}).")
    n_inputs = controls.size // n_phases
    controls = controls.reshape((n_inputs, n_phases))
    data['n_inputs'] = n_inputs

    # Ensure phases_duration is a 1D array
    data['phases_duration'] = np.asarray(data['phases_duration']).ravel()
    # print(data)

    return data

if __name__ == "__main__":
    data_file = "example_1_paper_NAHS.mat"
    data = load_data(data_file)
    
    optimizer = "adam"
    params_optimization(optimizer)

