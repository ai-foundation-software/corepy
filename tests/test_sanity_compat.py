import platform
import sys

import corepy


def test_python_version():
    """Ensure we are running on a supported Python version (>= 3.9)."""
    major, minor = sys.version_info[:2]
    print(f"Running on Python {major}.{minor}")
    assert (major, minor) >= (3, 9), f"Python version {major}.{minor} is too old!"


def test_platform_info():
    """Verify platform detection works and lists supported OSes."""
    system = platform.system()
    print(f"Platform: {sys.platform} / System: {system}")
    # We expect one of these, but we don't strictly fail if it's something else (e.g. some other Unix),
    # unless we want to be strict. For now, just logging it is fine, but let's assert it's a "known" one.
    assert system in ["Linux", "Darwin", "Windows"]


def test_corepy_import():
    """Ensure corepy can be imported and has version metadata."""
    assert corepy.__version__ is not None
    print(f"Corepy Version: {corepy.__version__}")
