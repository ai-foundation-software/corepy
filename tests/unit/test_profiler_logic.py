import importlib
import json
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

import corepy.profiler.core


class TestProfilerLogic:
    @pytest.fixture(autouse=True)
    def setup_profiler(self):
        import corepy

        # Save original state
        self.original_sys_mod = sys.modules.get("corepy._corepy_rust")
        self.original_attr = getattr(corepy, "_corepy_rust", None)
        self.had_attr = hasattr(corepy, "_corepy_rust")

        # Nuke from sys.modules
        # We use patch.dict to ensure it's removed or replaced
        # To trigger ImportError, we can remove it from sys.modules so it tries to load,
        # AND make the loader fail.
        # OR: we can set it to a dummy that raises ImportError on getattr? No.

        # The most reliable way for "from .. import X" to fail is if X is NOT in sys.modules
        # AND lookup in package fails.

        with patch.dict(sys.modules):
            if "corepy._corepy_rust" in sys.modules:
                del sys.modules["corepy._corepy_rust"]

            # Also remove from corepy module
            if self.had_attr:
                delattr(corepy, "_corepy_rust")

            # Now we must ensure that any ATTEMPT to reload it fails.
            # We can mock builtins.__import__ but that's global.
            # Or we can patch sys.meta_path to fail finding it?

            # Simpler: corepy.profiler.core does "from .. import _corepy_rust".
            # If we just removed it from corepy, it might try to import it again.
            # We need to make that import fail.

            # We can register a finder that raises ImportError for 'corepy._corepy_rust'
            class FailFinder:
                def find_spec(self, fullname, path, target=None):
                    if fullname == "corepy._corepy_rust":
                        # Return None -> not found? Or raise ImportError?
                        # Finder returns spec or None. If None, continues.
                        return None
                    return None

            # Attempting to make import fail by setting sys.modules[name] = None
            # In Python 3, this is an indication that the module is NOT found?
            # Actually, confusingly, it might just return None.

            # Let's try setting it to a mock that RAISES ImportError when accessed?
            # No, import just assigns names.

            # Let's try: sys.modules['corepy._corepy_rust'] = None
            # And reloading corepy.profiler.core
            sys.modules["corepy._corepy_rust"] = None
            # Note: This might cause "ModuleNotFoundError" or "ImportError".
            # The code catches "ImportError".

            try:
                importlib.reload(corepy.profiler.core)
            except (ImportError, ModuleNotFoundError):
                # If reload itself fails (it shouldn't, it should just fail the inner import and catch it)
                # But corepy/profiler/core.py catches ImportError.
                pass

            self.profiler_module = corepy.profiler.core

            yield

        # Restore
        if self.had_attr:
            corepy._corepy_rust = self.original_attr
        # sys.modules restored by patch.dict context? No, we yielded inside it?
        # Use explicit restore for safety if logic was complex.
        # Actually we yielded inside patch.dict(sys.modules), so sys.modules changes are reverted.
        # But corepy attr change is NOT.
        importlib.reload(corepy.profiler.core)

    def test_start_stop_clear_fallback(self):
        pm = self.profiler_module
        # If fallback worked, _RUST_AVAILABLE should be False
        # If sys.modules[..]=None caused "from .. import" to see None, then it didn't raise ImportError
        # and _RUST_AVAILABLE might be True but _enable_profiling is bound to None?

        # Verify state
        if pm._RUST_AVAILABLE:
            pytest.fail(
                f"Failed to force fallback. _RUST_AVAILABLE is True. _enable_profiling={pm._enable_profiling}"
            )

        pm.enable_profiling()
        assert pm._python_profiler.enabled

        pm.record_op("test_op", 10.0)
        report = json.loads(pm.profile_report(format="json"))
        assert "test_op" in report["operations"]

        pm.disable_profiling()
        assert not pm._python_profiler.enabled

        pm.clear_profile()
        assert pm._python_profiler.operations == {}

    def test_context_manager(self):
        pm = self.profiler_module
        pm.enable_profiling()
        with pm.ProfileContext("ctx1"):
            pm.record_op("op1", 5.0)

        report = json.loads(pm.profile_report(format="json"))
        # Debug print if fails
        print(report)
        op = report["operations"]["op1"]
        assert op["context"] == "ctx1"

    def test_export_formats(self):
        pm = self.profiler_module
        pm.enable_profiling()
        pm.record_op("op1", 5.0)

        # JSON
        with patch("builtins.open", mock_open()) as mock_file:
            pm.export_profile("test.json", format="json")
            mock_file().write.assert_called()

        # CSV
        with patch("builtins.open", mock_open()) as mock_file:
            pm.export_profile("test.csv", format="csv")
            mock_file().write.assert_called()

        # Chrome Tracing
        with patch("builtins.open", mock_open()) as mock_file:
            pm.export_profile("trace.json", format="chrome_tracing")
            mock_file().write.assert_called()

    def test_analysis_functions(self):
        pm = self.profiler_module
        pm.enable_profiling()
        # 100ms total
        pm.record_op("slow_op", 80.0)
        pm.record_op("fast_op", 20.0)

        # Detect bottlenecks (>20%)
        bottlenecks = pm.detect_bottlenecks(threshold=0.5)  # >50%
        assert len(bottlenecks) == 1
        assert bottlenecks[0]["operation"] == "slow_op"

        # Recommendations
        # Manually record to bypass backend default 'cpu' in record_op wrapper if needed?
        # Actually record_op defaults to 'cpu'.
        # The logic expects "CPU" (case sensitive likely).
        pm.record_op("matmul", 50.0, backend="CPU")

        recs = pm.get_recommendations()
        assert any(r["title"] == "Enable GPU for Matrix Multiplication" for r in recs)

    def test_regressions(self):
        pm = self.profiler_module
        pm.enable_profiling()
        pm.record_op("op1", 20.0)

        baseline = {"operations": {"op1": {"avg_time_ms": 10.0}}}

        regressions = pm.detect_regressions(baseline, threshold=1.5)
        assert len(regressions) == 1
        assert regressions[0]["operation"] == "op1"
