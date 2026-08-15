from corepy import array, ndarray
from corepy.backend import BackendType


def test_array_creation_defaults():
    t = array([1.0, 2.0, 3.0])
    assert t.backend == BackendType.CPU
    assert t.shape == (3,)


def test_array_auto_gpu_threshold_mock(monkeypatch):
    """
    Test that large arrays default to GPU if GPU is 'detected'.
    """
    # Mock detection to simulate GPU presence
    from corepy.backend.device import DeviceInfo

    mock_info = DeviceInfo(cpu_cores=4, gpu_count=1, platform_system="Linux")

    from corepy.backend import session

    # Reset singleton manually
    old_session = session._session
    try:
        session._session = None
        session.Session._instance = None  # Also reset class instance tracking if used

        # Apply patch to detect_devices BEFORE creating new session
        with monkeypatch.context() as m:
            m.setattr("corepy.backend.device.detect_devices", lambda: mock_info)
            m.setattr("corepy.backend.session.detect_devices", lambda: mock_info)

            # Block Rust module to test legacy fallback logic cleanly
            import sys

            m.setitem(sys.modules, "corepy._corepy_rust", None)

            # Re-initialize session (will trigger detect_devices)
            s = session.Session()
            session._session = s

            # 1. Small ndarray -> CPU
            t_small = array([1.0] * 1000)
            assert t_small.backend == BackendType.CPU

            # 2. Large ndarray -> GPU
            # THRESHOLD is 2,000_000. Legacy threshold triggered since Rust is unimported.
            t_large = array([1.0] * 2_000_001)
            assert t_large.backend == BackendType.CUDA
    finally:
        # Restore session
        session._session = old_session


def test_array_explicit_override_api(monkeypatch):
    # Mock GPU presence
    from corepy.backend.device import DeviceInfo

    gpu_info = DeviceInfo(cpu_cores=4, gpu_count=1, platform_system="Linux")

    from corepy.backend import session

    with monkeypatch.context() as m:
        m.setattr("corepy.backend.session.detect_devices", lambda: gpu_info)
        m.setattr("corepy.backend.device.detect_devices", lambda: gpu_info)

        # Reset session
        old_session_var = session._session
        old_session_instance = session.Session._instance
        session._session = None
        session.Session._instance = None
        try:
            t = ndarray([1, 2, 3], backend="cuda")
            assert t.backend == BackendType.CUDA
        finally:
            session._session = old_session_var
            session.Session._instance = old_session_instance


def test_array_explicit_device_api(monkeypatch):
    # Mock GPU presence for "cuda:0" request
    from corepy.backend.device import DeviceInfo

    gpu_info = DeviceInfo(cpu_cores=4, gpu_count=1, platform_system="Linux")

    from corepy.backend import session

    with monkeypatch.context() as m:
        m.setattr("corepy.backend.session.detect_devices", lambda: gpu_info)
        m.setattr("corepy.backend.device.detect_devices", lambda: gpu_info)

        old_session_var = session._session
        old_session_instance = session.Session._instance
        session._session = None
        session.Session._instance = None
        try:
            t = array([1, 2, 3], device="cuda:0")
            assert t.backend == BackendType.CUDA
        finally:
            session._session = old_session_var
            session.Session._instance = old_session_instance
