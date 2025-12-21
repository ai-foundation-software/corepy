import pytest
import corepy

def test_version_exists():
    assert hasattr(corepy, "__version__")
    assert isinstance(corepy.__version__, str)
    assert len(corepy.__version__) > 0

def test_public_api_attributes():
    """Ensure core submodules are exposed."""
    assert hasattr(corepy, "data")
    assert hasattr(corepy, "schema")
    assert hasattr(corepy, "runtime")

def test_cpp_extension_add_one():
    """Verify the C++ extension function is available and works."""
    # This assumes the build environment is correct (which it is now)
    try:
        from corepy import add_one
        assert add_one(5) == 6
    except ImportError:
        pytest.fail("C++ extension `add_one` could not be imported. Build likely failed.")
