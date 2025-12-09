import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwiLin
from src import StochasticGradientDescent, RMSPropOptimizer, AdamOptimizer
from src import sgd_optimize, rmsprop_optimize, adam_optimize
from datetime import datetime
import pathlib

def switched_problem(n_phases=5):
    """
    Set up a switched linear problem and compute cost and gradient functions
    """
    model = {
        'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
        'B': [np.array([[0.25], [2], [0]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 10
    
    x0 = np.array([1.3440, -4.5850, 5.6470])
    
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

    Q = 1. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))


    swi_lin.precompute_matrices(x0, Q, R, P)
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
        params = np.asarray(params)
        # From a single flattened params vector, unpack into controls and durations
        u = params[:n_phases * n_inputs].reshape((n_phases, n_inputs)).tolist()
        phases_duration = params[n_phases * n_inputs:].reshape((n_phases,)).tolist()
        params_list = u + phases_duration
        # Compute the cost function for a single example (no batch)
        J = float(J_func(*params_list).full().item())
        
        if data is not None:
            u = data['controls'].ravel()
            phases_duration = data['phases_duration'].ravel()
            params_ref = np.concatenate([u, phases_duration])
            # print(f"Reference params: {params_ref}")
            
            # Compute the loss wrt reference params
            params_ref = np.asarray(params_ref)
            # add numpy sum of squared differences to scalar loss
            J += float(np.sum((params - params_ref) ** 2))

        return J

    def gradient_function(params, indices=None, data=None):
        """
        params: 1D array (n_params,) or 2D array (batch_size, n_params)
        indices: optional indices to select a minibatch from a 2D params array
        Returns gradient vector (n_params,) averaged over minibatch if batch provided.
        """
        params = np.asarray(params)
        # From a single flattened params vector, unpack into controls and durations
        u = params[:n_phases * n_inputs].reshape((n_phases, n_inputs)).tolist()
        phases_duration = params[n_phases * n_inputs:].reshape((n_phases,)).tolist()
        params_list = u + phases_duration
        # single example
        grad_J = np.asarray(grad_J_func(*params_list).full().ravel())
        
        if data is not None:
            u = data['controls'].ravel()
            phases_duration = data['phases_duration'].ravel()
            params_ref = np.concatenate([u, phases_duration])
            params_ref = np.asarray(params_ref)
            # add gradient of the squared differences to grad_J
            grad_J += 2 * (params - params_ref)

        return grad_J


    return J_func, grad_J_func, cost_function, gradient_function

def params_optimization(optimizer="sgd", data=None):
    """
    Perform optimization using the switched linear problem setup.
    """
    n_phases = 80

    _, _, cost_function, gradient_function = switched_problem(n_phases)
    
    # If data is provided, build the initial parameters by perturbating the reference
    if  None:
        u = data['controls']
        phases_duration = data['phases_duration']
        true_params = np.concatenate([u.ravel(), phases_duration.ravel()])
        # Add some noise to the initial parameters
        initial_params = np.random.normal(true_params, 0.1, true_params.shape)
        # Ensure phase durations are non-negative by flooring to a small positive value
        n_control_params = n_phases * data['n_inputs']
        controls_init = initial_params[:n_control_params].astype(float)
        durations_init = initial_params[n_control_params:].astype(float)
        eps = 1e-6
        durations_init = np.maximum(durations_init, eps)
        initial_params = np.concatenate([controls_init, durations_init])
        
    else:
        # initial inputs are zero, initial durations split the time_horizon evenly across phases
        n_inputs = 1
        time_horizon = 10.0  # same horizon used in switched_problem
        controls_init = np.zeros(n_phases * n_inputs, dtype=float)
        durations_init = np.full(n_phases, time_horizon / float(n_phases), dtype=float)
        initial_params = np.concatenate([controls_init, durations_init])
        
    # Create a mask matrix to extract delta parameters
    n_control_params = n_phases * (data['n_inputs'] if data is not None else 1)
    n_params = n_control_params + n_phases
    delta_mask = np.zeros(n_params, dtype=float)
    delta_mask[n_control_params:] = 1.0

    # Choose learning rate schedule
    schedules = [
        {'name': 'Constant', 'schedule': 'constant'},
        {'name': 'Step', 'schedule': 'step', 'params': {'step_size': 20, 'gamma': 0.5}},
        {'name': 'Exponential', 'schedule': 'exponential', 'params': {'decay_rate': 0.05}},
        {'name': 'Inverse', 'schedule': 'inverse', 'params': {'decay_rate': 0.1}},
    ]
    schedule = schedules[0]['schedule']
    params = schedules[0].get('params', {})

    if optimizer == "sgd":
        params_optimized, history = sgd_optimize(
                params_init=initial_params,
                gradient_func=gradient_function,
                loss_func=cost_function,  
                learning_rate=0.001,
                n_epochs=1000,
                learning_rate_schedule=schedule,
                schedule_params=params,
                data=data,
        )
    elif optimizer == "rmsprop":
        params_optimized, history = rmsprop_optimize(
                params_init=initial_params,
                gradient_func=gradient_function,
                loss_func=cost_function,  
                learning_rate=0.001,
                n_epochs=1000,
                learning_rate_schedule=schedule,
                schedule_params=params,
                weight_decay=0.9,
                eps=1e-8,
                data=data,
        )
    elif optimizer == "adam":
        params_optimized, history = adam_optimize(
                params_init=initial_params,
                gradient_func=gradient_function,
                loss_func=cost_function,
                learning_rate=0.001,
                n_epochs=1000,
                learning_rate_schedule=schedule,
                schedule_params=params,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                data=data,
                delta_mask=delta_mask,
                time_horizon=time_horizon,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Print results
    # print("Optimized Parameters:", params_optimized)
    n_inputs = data['n_inputs'] if data is not None else 1
    plot_params(params_optimized, n_phases, n_inputs, save_figure=False)
    plot_history(history, save_figure=False)
    
def plot_history(history, save_figure=False):
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
    
    if save_figure:

        # Prefer an 'image' (or 'images') folder inside the package (search up from this file)
        pkg_dir = pathlib.Path(__file__).resolve().parent
        image_dir = None
        for p in [pkg_dir] + list(pkg_dir.parents):
            for name in ('image', 'images'):
                cand = p / name
            if cand.exists() and cand.is_dir():
                image_dir = cand
                break
            if image_dir is not None:
                break

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname_base = f"optimization_history_{timestamp}"
        png_path = image_dir / (fname_base + ".png")
        # svg_path = image_dir / (fname_base + ".svg")

        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        # fig.savefig(str(svg_path), bbox_inches='tight')
    else:
        plt.show()
    
def plot_params(params_optimized, n_phases, n_inputs=1, save_figure=False):
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
    
    time_horizon = np.sum(durations)
    tgrid = []
    points = 1000
    time = 0
    next_time = 0
    for i in range(n_phases):
        next_time = next_time + durations[i]
        tgrid = np.concatenate((tgrid, np.linspace(time, next_time, points, endpoint=False)))
        time = time + durations[i]
    tgrid = np.concatenate((tgrid, [time_horizon]))
    edges = np.concatenate([[0], np.cumsum(durations)])

    # Create subplots: n_inputs plots for controls + 1 for durations
    fig, axes = plt.subplots(n_inputs + 1, 1, figsize=(10, 3 * (n_inputs + 1)), sharex=True)
    
    # If only one subplot, axes is not an array
    if n_inputs == 0:
        axes = [axes]

    # Plot each control input
    colors = ['tab:orange', 'tab:cyan', 'tab:green', 'tab:red']
    for i in range(n_inputs):
        axes[i].step(edges[:-1], controls[:, i], where='post',
                     color=colors[i % len(colors)], lw=2)
        axes[i].set_ylabel(f'Control Input {i+1}')
        axes[i].set_title(f'Optimized Control Input {i+1} over Phases')
        axes[i].grid(True)
        axes[i].set_xlim([0, edges[-1]])

    # Plot durations as bars
    axes[n_inputs].bar(edges[:-1], durations, width=durations, align='edge',
                      color='tab:purple', alpha=0.6, edgecolor='black')
    axes[n_inputs].set_ylabel('Phase Duration')
    axes[n_inputs].set_xlabel('Time (s)')
    axes[n_inputs].set_title('Optimized Phase Durations over Phases')
    axes[n_inputs].grid(True)
    axes[n_inputs].set_xlim([0, edges[-1]])

    if save_figure:

        # Prefer an 'image' (or 'images') folder inside the package (search up from this file)
        pkg_dir = pathlib.Path(__file__).resolve().parent
        image_dir = None
        for p in [pkg_dir] + list(pkg_dir.parents):
            for name in ('image', 'images'):
                cand = p / name
            if cand.exists() and cand.is_dir():
                image_dir = cand
                break
            if image_dir is not None:
                break

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname_base = f"params_nph{n_phases}_ninputs{n_inputs}_{timestamp}"
        png_path = image_dir / (fname_base + ".png")
        # svg_path = image_dir / (fname_base + ".svg")

        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        # fig.savefig(str(svg_path), bbox_inches='tight')
    else:
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
    # print(data)

    return data

    # return data
if __name__ == "__main__":
    data_file = "optimal_params.mat"
    data = load_data(data_file)
    
    optimizer = "adam"
    params_optimization(optimizer, data)

