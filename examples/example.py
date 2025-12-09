"""
Example usage of the Stochastic Gradient Descent implementation.

This script demonstrates how to use the SGD optimizer with your own
gradient expressions.
"""

import numpy as np
import matplotlib.pyplot as plt
from src import StochasticGradientDescent, sgd_optimize


def example_quadratic():
    """
    Example 1: Minimize a simple quadratic function.
    f(x) = (x - 3)^T @ A @ (x - 3)
    where A is a positive definite matrix.
    """
    print("=" * 60)
    print("Example 1: Quadratic Function Minimization")
    print("=" * 60)
    
    # Define the problem
    n_dim = 5
    np.random.seed(42)
    A = np.random.randn(n_dim, n_dim)
    A = A.T @ A + np.eye(n_dim)  # Make positive definite
    x_opt = 3 * np.ones(n_dim)  # Optimal point
    
    # Loss function
    def loss_func(x, data=None):
        diff = x - x_opt
        return diff.T @ A @ diff
    
    # Gradient function (this is what you would get from your gradient expressions)
    def gradient_func(x, indices=None, data=None):
        diff = x - x_opt
        return 2 * A @ diff
    
    # Initial parameters
    x_init = np.random.randn(n_dim)
    
    # Run SGD
    x_final, history = sgd_optimize(
        params_init=x_init,
        gradient_func=gradient_func,
        loss_func=loss_func,
        learning_rate=0.01,
        n_epochs=100,
        verbose=True
    )
    
    print(f"\nOptimal point: {x_opt}")
    print(f"Found point:   {x_final}")
    print(f"Distance from optimal: {np.linalg.norm(x_final - x_opt):.6e}")
    
    # Plot convergence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(history['gradient_norm'])
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm vs Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sgd_quadratic_convergence.png', dpi=150)
    print("\nConvergence plot saved as 'sgd_quadratic_convergence.png'")


def example_linear_regression():
    """
    Example 2: Linear regression with mini-batch SGD.
    """
    print("\n" + "=" * 60)
    print("Example 2: Linear Regression with Mini-batch SGD")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    # Loss function (MSE)
    def loss_func(w, data=None):
        X, y = data
        predictions = X @ w
        return np.mean((predictions - y) ** 2)
    
    # Gradient function
    def gradient_func(w, indices=None, data=None):
        X, y = data
        if indices is not None:
            X_batch = X[indices]
            y_batch = y[indices]
        else:
            X_batch = X
            y_batch = y
        
        predictions = X_batch @ w
        gradient = 2 * X_batch.T @ (predictions - y_batch) / len(y_batch)
        return gradient
    
    # Initial parameters
    w_init = np.zeros(n_features)
    
    # Run SGD with different configurations
    configs = [
        {'name': 'SGD', 'momentum': 0.0, 'batch_size': 32},
        {'name': 'SGD + Momentum', 'momentum': 0.9, 'batch_size': 32},
        {'name': 'SGD + Nesterov', 'momentum': 0.9, 'nesterov': True, 'batch_size': 32},
    ]
    
    results = {}
    
    for config in configs:
        name = config.pop('name')
        print(f"\n--- Running {name} ---")
        
        w_final, history = sgd_optimize(
            params_init=w_init.copy(),
            gradient_func=gradient_func,
            loss_func=loss_func,
            learning_rate=0.1,
            n_epochs=50,
            data=(X, y),
            n_samples=n_samples,
            verbose=False,
            **config
        )
        
        results[name] = history
        print(f"Final loss: {history['loss'][-1]:.6e}")
        print(f"Parameter error: {np.linalg.norm(w_final - w_true):.6e}")
    
    # Plot comparison
    plt.figure(figsize=(10, 5))
    for name, history in results.items():
        plt.semilogy(history['loss'], label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Comparison of SGD Variants')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sgd_comparison.png', dpi=150)
    print("\nComparison plot saved as 'sgd_comparison.png'")


def example_learning_rate_schedules():
    """
    Example 3: Comparison of learning rate schedules.
    """
    print("\n" + "=" * 60)
    print("Example 3: Learning Rate Schedules")
    print("=" * 60)
    
    # Simple quadratic problem
    n_dim = 10
    np.random.seed(42)
    A = np.random.randn(n_dim, n_dim)
    A = A.T @ A + np.eye(n_dim)
    x_opt = np.ones(n_dim)
    
    def loss_func(x, data=None):
        diff = x - x_opt
        return diff.T @ A @ diff
    
    def gradient_func(x, indices=None, data=None):
        diff = x - x_opt
        return 2 * A @ diff
    
    x_init = np.random.randn(n_dim)
    
    schedules = [
        {'name': 'Constant', 'schedule': 'constant'},
        {'name': 'Step', 'schedule': 'step', 'params': {'step_size': 20, 'gamma': 0.5}},
        {'name': 'Exponential', 'schedule': 'exponential', 'params': {'decay_rate': 0.05}},
        {'name': 'Inverse', 'schedule': 'inverse', 'params': {'decay_rate': 0.1}},
    ]
    
    results = {}
    
    for config in schedules:
        name = config['name']
        schedule = config['schedule']
        params = config.get('params', {})
        
        print(f"\n--- {name} Schedule ---")
        
        _, history = sgd_optimize(
            params_init=x_init.copy(),
            gradient_func=gradient_func,
            loss_func=loss_func,
            learning_rate=0.1,
            n_epochs=100,
            learning_rate_schedule=schedule,
            schedule_params=params,
            verbose=False
        )
        
        results[name] = history
        print(f"Final loss: {history['loss'][-1]:.6e}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, history in results.items():
        ax1.semilogy(history['loss'], label=name, linewidth=2)
        ax2.plot(history['learning_rate'], label=name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('sgd_schedules.png', dpi=150)
    print("\nSchedules comparison saved as 'sgd_schedules.png'")


def example_with_custom_gradients():
    """
    Example 4: Template for using your own gradient expressions.
    
    This shows how to integrate SGD with gradient expressions you've
    computed elsewhere (e.g., from symbolic differentiation, automatic
    differentiation, or analytical derivations).
    """
    print("\n" + "=" * 60)
    print("Example 4: Template for Custom Gradients")
    print("=" * 60)
    
    print("""
    To use SGD with your own gradient expressions:
    
    1. Define your gradient function with this signature:
       
       def my_gradient_func(params, indices=None, data=None):
           '''
           params: current parameter values (np.ndarray)
           indices: mini-batch indices (optional, for stochastic gradient)
           data: any additional data needed (optional)
           '''
           # Use your gradient expressions here
           # gradient = ... (your gradient computation)
           return gradient
    
    2. Optionally define a loss function for monitoring:
       
       def my_loss_func(params, data=None):
           # loss = ... (your loss computation)
           return loss
    
    3. Run SGD:
       
       params_optimized, history = sgd_optimize(
           params_init=initial_params,
           gradient_func=my_gradient_func,
           loss_func=my_loss_func,  # optional
           learning_rate=0.01,
           n_epochs=100,
           # ... other parameters
       )
    
    For switched OCP problems, your gradient_func would compute
    gradients with respect to control inputs and mode durations.
    """)


if __name__ == "__main__":
    # Run all examples
    example_quadratic()
    example_linear_regression()
    example_learning_rate_schedules()
    example_with_custom_gradients()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    
    plt.show()
