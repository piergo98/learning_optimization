import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from ocslc.switched_linear_mpc import SwitchedLinearMPC
from sgd import StochasticGradientDescent, sgd_optimize

def switched_problem():
    """
    Set up a switched linear problem and compute cost and gradient functions
    """
    model = {
        'A': [np.zeros((2, 2))],
        'B': [np.eye(2)],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 2
    time_horizon = 10

    x0 = np.array([5, 2])
    
    xr = np.array([1, -3])
    
    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_phases, 
        time_horizon, 
        auto=False, 
        multiple_shooting=False, 
        x0=x0
    )

    Q = 0. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
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
        # single example
        if params.ndim == 1:
            return float(J_func(*params).full().item())

        # batch case
        batch = params
        if indices is not None:
            batch = batch[indices]
        vals = [float(J_func(*p).full().item()) for p in batch]
        return float(np.mean(vals))

    def gradient_function(params, indices=None, data=None):
        """
        params: 1D array (n_params,) or 2D array (batch_size, n_params)
        indices: optional indices to select a minibatch from a 2D params array
        Returns gradient vector (n_params,) averaged over minibatch if batch provided.
        """
        params = np.asarray(params)
        # single example
        if params.ndim == 1:
            return np.asarray(grad_J_func(*params).full().ravel())

        # batch case
        batch = params
        if indices is not None:
            batch = batch[indices]
        grads = [np.asarray(grad_J_func(*p).full().ravel()) for p in batch]
        return np.mean(grads, axis=0)

    return J_func, grad_J_func, cost_function, gradient_function

def params_optimization():
    """
    Perform optimization using the switched linear problem setup.
    """

    _, _, cost_function, gradient_function = switched_problem()
    
    initial_params = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # Example initial guess
    
    # Run SGD with different configurations
    configs = [
        {'momentum': 0.0, 'batch_size': 32},
        {'momentum': 0.9, 'batch_size': 32},
        {'momentum': 0.9, 'nesterov': True, 'batch_size': 32},
    ]
    
    schedules = [
        {'name': 'Constant', 'schedule': 'constant'},
        {'name': 'Step', 'schedule': 'step', 'params': {'step_size': 20, 'gamma': 0.5}},
        {'name': 'Exponential', 'schedule': 'exponential', 'params': {'decay_rate': 0.05}},
        {'name': 'Inverse', 'schedule': 'inverse', 'params': {'decay_rate': 0.1}},
    ]
    schedule = schedules[0]['schedule']
    params = schedules[0].get('params', {})
    
    params_optimized, history = sgd_optimize(
            params_init=initial_params,
            gradient_func=gradient_function,
            loss_func=cost_function,  # optional
            learning_rate=0.01,
            n_epochs=1000,
            learning_rate_schedule=schedule,
            schedule_params=params,
            **configs[2]
    )
    
    # Print results
    print("Optimized Parameters:", params_optimized)
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

if __name__ == "__main__":
    params_optimization()

