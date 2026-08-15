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
