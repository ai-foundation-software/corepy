import sys
import pytest
from unittest.mock import patch
import importlib

def test_init_fallback_missing_extension():
    """
    Test that corepy imports gracefully even if _corepy_cpp is missing,
    and that the fallback add_one function raises ImportError.
    """
    # 1. Save state
    original_modules = sys.modules.copy()
    
    try:
        # 2. Unload corepy modules to force strict re-import
        for key in list(sys.modules.keys()):
            if key.startswith("corepy"):
                del sys.modules[key]
        
        # 3. Simulate invalid implementation for the extension
        # Setting it to None in sys.modules causes ModuleNotFoundError on import in Python >= 3.
        # This catches ImportError in the __init__.py try/except block.
        sys.modules["corepy._corepy_cpp"] = None
        
        # 4. Re-import corepy
        import corepy
        importlib.reload(corepy)
        
        # 5. Verify proper version (should still be set)
        assert corepy.__version__ == "0.2.0"
        
        # 6. Verify fallback behavior
        # add_one should exist (as the python fallback)
        assert hasattr(corepy, "add_one")
        
        # 7. Verify calling it raises ImportError
        with pytest.raises(ImportError, match="C\\+\\+ extension not loaded"):
            corepy.add_one(10)

    finally:
        # 8. Restore original state to not break other tests
        # We need to aggressively clean up our mock entries
        if "corepy._corepy_cpp" in sys.modules and sys.modules["corepy._corepy_cpp"] is None:
             del sys.modules["corepy._corepy_cpp"]
             
        # Restore old modules
        sys.modules.update(original_modules)
        
        # Force reload of corepy to ensure it's in a good state for subsequent tests (if any)
        # Note: If subsequent tests run in the same process, we want the REAL corepy back.
        if "corepy" in sys.modules:
             import corepy
             importlib.reload(corepy)
