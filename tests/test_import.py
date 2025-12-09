"""Test package initialization."""


def test_package_metadata():
    """Test that package has proper metadata."""
    import src
    
    assert hasattr(src, '__version__')
    assert isinstance(src.__version__, str)
    assert len(src.__version__.split('.')) >= 2  # At least major.minor
    
    # Check __all__ is defined
    assert hasattr(src, '__all__')
    assert isinstance(src.__all__, list)
    assert len(src.__all__) > 0
