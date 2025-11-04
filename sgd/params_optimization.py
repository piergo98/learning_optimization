import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwitchedLinearMPC
from sgd import StochasticGradientDescent, RMSPropOptimizer, AdamOptimizer
from sgd import sgd_optimize, rmsprop_optimize, adam_optimize

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
    
    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_phases, 
        time_horizon, 
        auto=False, 
        multiple_shooting=False, 
        x0=x0
    )

    Q = 1. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    swi_lin_mpc.set_cost_function_single_shooting(R, x0)
    J = swi_lin_mpc.cost
    J_func = ca.Function('J_func', [*ca.symvar(J)], [J])
        
    grad_J_u = []
    grad_J_delta = []

    for k in range(n_phases):
        # Compute gradient of the cost
        du, d_delta = swi_lin_mpc.grad_cost_function(k, R)
        # print(f"Phase {k}: du = {du}, d_delta = {d_delta}")

        grad_J_u.append(du)
        grad_J_delta.append(d_delta)

    # Stack the gradients for optimization
    # Build interleaved gradient: U_0 Delta_0 U_1 Delta_1 ...
    interleaved = []
    for k in range(n_phases):
        u_k = np.ravel(grad_J_u[k])
        delta_k = np.atleast_1d(grad_J_delta[k]).ravel()
        interleaved.append(u_k)
        interleaved.append(delta_k)

    grad_J = np.hstack(interleaved)
    # print(f"Gradient shape: {grad_J.shape}")

    # keep the original stacked forms if needed
    grad_J_u = np.hstack(grad_J_u)
    grad_J_delta = np.hstack(grad_J_delta)

    # Create a CasADi function for the gradient
    grad_J_func = ca.Function('grad_J', [*ca.symvar(grad_J)], [grad_J])
    
    # Create wrapper functions for the optimizer
    def cost_function(params, indices=None, data=None):
        """
        params: 1D array (n_params,) or 2D array (batch_size, n_params)
        indices: optional indices to select a minibatch from a 2D params array
        Returns the scalar loss (averaged over minibatch if batch provided).
        """
        params = np.asarray(params)
        # Compute the cost function for a single example (no batch)
        J = float(J_func(*params).full().item())
        
        if data is not None:
            u = data['controls']
            phases_duration = data['phases_duration']
            params_ref = np.concatenate([u.ravel(), phases_duration.ravel()])
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
        # single example
        grad_J = np.asarray(grad_J_func(*params).full().ravel())
        
        if data is not None:
            u = data['controls']
            phases_duration = data['phases_duration']
            params_ref = np.concatenate([u.ravel(), phases_duration.ravel()])
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
    if data is not None:
        u = data['controls']
        phases_duration = data['phases_duration']
        initial_params = np.concatenate([u.ravel(), phases_duration.ravel()])
        # Add some noise to the initial parameters
        initial_params += np.random.normal(0, 0.1, initial_params.shape)
    else:
        initial_params = np.zeros(n_phases * (1 + 1))  # 1 inputs + 1 duration per phase

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
                loss_func=cost_function,  # optional
                learning_rate=0.0001,
                n_epochs=1000,
                learning_rate_schedule=schedule,
                schedule_params=params,
                data=data,
        )
    elif optimizer == "rmsprop":
        params_optimized, history = rmsprop_optimize(
                params_init=initial_params,
                gradient_func=gradient_function,
                loss_func=cost_function,  # optional
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
                loss_func=cost_function,  # optional
                learning_rate=0.01,
                n_epochs=1000,
                learning_rate_schedule=schedule,
                schedule_params=params,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                data=data,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Print results
    print("Optimized Parameters:", params_optimized)
    plot_params(params_optimized, n_phases)
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
    
def plot_params(params_optimized, n_phases):
    """
    Plot the optimized parameters (inputs and durations) over phases.
    """
    inputs = params_optimized[0::2]
    durations = params_optimized[1::2]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Inputs
    axes[0].step(range(n_phases), inputs, where='post', color='tab:orange', lw=2)
    axes[0].set_ylabel('Control Input')
    axes[0].set_title('Optimized Control Inputs over Phases')
    axes[0].grid(True)

    # Durations
    axes[1].step(range(n_phases), durations, where='post', color='tab:purple', lw=2)
    axes[1].set_ylabel('Phase Duration')
    axes[1].set_xlabel('Phase')
    axes[1].set_title('Optimized Phase Durations over Phases')
    axes[1].grid(True)

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
    # print(data)

    return data

    # return data
if __name__ == "__main__":
    data_file = "optimal_params.mat"
    data = load_data(data_file)
    
    optimizer = "rmsprop"
    params_optimization(optimizer, data)

