"""
Basic tests for the optimizers package.
Run with: pytest tests/
"""
import numpy as np
import pytest


def test_import_optimizers():
    """Test that the optimizers package can be imported."""
    import src
    assert hasattr(src, '__version__')
    assert hasattr(src, 'sgd_optimize')
    assert hasattr(src, 'StochasticGradientDescent')


def test_sgd_optimize_simple():
    """Test sgd_optimize on a simple quadratic problem."""
    from src import sgd_optimize
    
    # Minimize ||x - target||^2
    target = np.array([1.0, 2.0, 3.0])
    
    def gradient_func(params, indices=None, data=None):
        return 2 * (params - target)
    
    def loss_func(params, data=None):
        return np.sum((params - target) ** 2)
    
    params_init = np.zeros(3)
    params_opt, history = sgd_optimize(
        params_init=params_init,
        gradient_func=gradient_func,
        loss_func=loss_func,
        optimizer='sgd',
        learning_rate=0.1,
        n_epochs=100,
        verbose=False
    )
    
    # Check convergence
    assert np.allclose(params_opt, target, atol=0.1)
    assert len(history['loss']) > 0
    assert history['loss'][-1] < history['loss'][0]  # Loss should decrease


def test_adam_optimize():
    """Test adam_optimize on a simple problem."""
    from src import adam_optimize
    
    target = np.array([5.0, -3.0])
    
    def gradient_func(params, indices=None, data=None):
        return 2 * (params - target)
    
    params_init = np.zeros(2)
    params_opt, history = adam_optimize(
        params_init=params_init,
        gradient_func=gradient_func,
        learning_rate=0.1,
        n_epochs=50,
        verbose=False
    )
    
    assert np.allclose(params_opt, target, atol=0.5)


def test_rmsprop_optimize():
    """Test rmsprop_optimize on a simple problem."""
    from src import rmsprop_optimize
    
    target = np.array([2.0, -1.0, 0.5])
    
    def gradient_func(params, indices=None, data=None):
        return 2 * (params - target)
    
    params_init = np.zeros(3)
    params_opt, history = rmsprop_optimize(
        params_init=params_init,
        gradient_func=gradient_func,
        learning_rate=0.1,
        n_epochs=50,
        verbose=False
    )
    
    assert np.allclose(params_opt, target, atol=0.5)


def test_torch_available_flag():
    """Test that _TORCH_AVAILABLE flag is set correctly."""
    import src
    assert isinstance(src._TORCH_AVAILABLE, bool)


@pytest.mark.skipif(
    not hasattr(__import__('optimizers'), 'gpu_optimize'),
    reason="PyTorch not installed"
)
def test_gpu_optimize_cpu():
    """Test GPU optimize on CPU (if torch available)."""
    from src import gpu_optimize
    import torch
    
    target = torch.tensor([1.0, 2.0])
    
    def gradient_func(params, indices=None, data=None):
        return 2 * (params - target)
    
    params_init = torch.zeros(2)
    params_opt, history = gpu_optimize(
        params_init=params_init,
        gradient_func=gradient_func,
        optimizer='adam',
        learning_rate=0.1,
        n_epochs=50,
        device='cpu',
        verbose=False
    )
    
    assert torch.allclose(params_opt, target, atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
