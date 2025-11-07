"""Test package initialization."""


def test_package_metadata():
    """Test that package has proper metadata."""
    import optimizers
    
    assert hasattr(optimizers, '__version__')
    assert isinstance(optimizers.__version__, str)
    assert len(optimizers.__version__.split('.')) >= 2  # At least major.minor
    
    # Check __all__ is defined
    assert hasattr(optimizers, '__all__')
    assert isinstance(optimizers.__all__, list)
    assert len(optimizers.__all__) > 0
