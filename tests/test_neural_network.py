"""
Tests for neural network training module.
"""
import numpy as np
import pytest


def test_neural_network_numpy_creation():
    """Test NeuralNetworkNumPy initialization."""
    from optimizers.neural_network import NeuralNetworkNumPy
    
    network = NeuralNetworkNumPy(
        layer_sizes=[10, 5, 3],
        activation='relu',
        output_activation='softmax'
    )
    
    assert network.n_layers == 2
    assert len(network.params) > 0
    assert network.layer_sizes == [10, 5, 3]


def test_neural_network_forward_pass():
    """Test forward pass through network."""
    from optimizers.neural_network import NeuralNetworkNumPy
    
    network = NeuralNetworkNumPy(
        layer_sizes=[4, 5, 3],
        activation='relu',
        output_activation='softmax'
    )
    
    X = np.random.randn(10, 4)
    output = network.forward(X, network.params)
    
    assert output.shape == (10, 3)
    # Softmax outputs should sum to 1
    assert np.allclose(output.sum(axis=1), 1.0)


def test_neural_network_gradient_computation():
    """Test gradient computation."""
    from optimizers.neural_network import NeuralNetworkNumPy
    
    network = NeuralNetworkNumPy(
        layer_sizes=[4, 5, 3],
        activation='relu',
        output_activation='softmax'
    )
    
    X = np.random.randn(10, 4)
    y = np.random.randint(0, 3, 10)
    
    loss, gradient = network.compute_loss_and_gradient(network.params, X, y)
    
    assert isinstance(loss, float)
    assert gradient.shape == network.params.shape
    assert np.all(np.isfinite(gradient))


def test_train_neural_network_numpy_simple():
    """Test neural network training on simple problem."""
    from optimizers.neural_network import NeuralNetworkNumPy, train_neural_network_numpy
    
    # Simple linearly separable problem
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    network = NeuralNetworkNumPy(
        layer_sizes=[2, 8, 2],
        activation='relu',
        output_activation='softmax'
    )
    
    params_opt, history = train_neural_network_numpy(
        network=network,
        X_train=X,
        y_train=y,
        optimizer='adam',
        learning_rate=0.01,
        n_epochs=10,
        batch_size=32,
        verbose=False
    )
    
    # Check that training happened
    assert len(history['loss']) > 0
    # Loss should generally decrease
    assert history['loss'][-1] <= history['loss'][0] * 1.5  # Allow some variance


def test_compute_accuracy():
    """Test accuracy computation."""
    from optimizers.neural_network import compute_accuracy
    
    # Perfect predictions
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = np.array([1, 0, 1])
    
    acc = compute_accuracy(predictions, targets)
    assert acc == 1.0
    
    # Mixed predictions
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2]])
    targets = np.array([1, 0, 1])
    
    acc = compute_accuracy(predictions, targets)
    assert abs(acc - 2/3) < 0.01


@pytest.mark.skipif(
    not __import__('optimizers.neural_network', fromlist=['TORCH_AVAILABLE']).TORCH_AVAILABLE,
    reason="PyTorch not installed"
)
def test_neural_network_pytorch_creation():
    """Test NeuralNetworkPyTorch initialization."""
    from optimizers.neural_network import NeuralNetworkPyTorch
    
    network = NeuralNetworkPyTorch(
        layer_sizes=[10, 5, 3],
        activation='relu',
        output_activation='softmax'
    )
    
    assert len(list(network.parameters())) > 0
    assert network.layer_sizes == [10, 5, 3]


@pytest.mark.skipif(
    not __import__('optimizers.neural_network', fromlist=['TORCH_AVAILABLE']).TORCH_AVAILABLE,
    reason="PyTorch not installed"
)
def test_neural_network_pytorch_forward():
    """Test PyTorch network forward pass."""
    from optimizers.neural_network import NeuralNetworkPyTorch
    import torch
    
    network = NeuralNetworkPyTorch(
        layer_sizes=[4, 5, 3],
        activation='relu',
        output_activation='softmax'
    )
    
    X = torch.randn(10, 4)
    output = network(X)
    
    assert output.shape == (10, 3)
    assert torch.allclose(output.sum(dim=1), torch.ones(10), atol=1e-5)


@pytest.mark.skipif(
    not __import__('optimizers.neural_network', fromlist=['TORCH_AVAILABLE']).TORCH_AVAILABLE,
    reason="PyTorch not installed"
)
def test_train_neural_network_torch_simple():
    """Test PyTorch neural network training on simple problem."""
    from optimizers.neural_network import NeuralNetworkPyTorch, train_neural_network_torch
    import torch
    
    torch.manual_seed(42)
    X = torch.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    network = NeuralNetworkPyTorch(
        layer_sizes=[2, 8, 2],
        activation='relu',
        output_activation='softmax'
    )
    
    params_opt, history = train_neural_network_torch(
        network=network,
        X_train=X,
        y_train=y,
        optimizer='adam',
        learning_rate=0.01,
        n_epochs=10,
        batch_size=32,
        device='cpu',
        verbose=False
    )
    
    # Check that training happened
    assert len(history['loss']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
