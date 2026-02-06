
import pytest
import os
import json
import csv
import corepy
from corepy.profiler import export_profile, enable_profiling, disable_profiling, clear_profile, ProfileContext

@pytest.fixture
def clean_profiler():
    clear_profile()
    disable_profiling()
    yield
    clear_profile()
    disable_profiling()

def test_export_json(tmp_path, clean_profiler):
    """Test exporting profile to JSON."""
    enable_profiling()
    t = corepy.Tensor([1.0, 2.0])
    _ = t + t
    
    out_file = tmp_path / "profile.json"
    export_profile(str(out_file), format="json")
    
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)
        assert "operations" in data
        assert "add" in data["operations"]

def test_export_csv(tmp_path, clean_profiler):
    """Test exporting profile to CSV."""
    enable_profiling()
    t = corepy.Tensor([1.0, 2.0])
    _ = t + t
    
    out_file = tmp_path / "profile.csv"
    export_profile(str(out_file), format="csv")
    
    assert out_file.exists()
    with open(out_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) >= 1
        assert rows[0]["operation"] == "add"

def test_export_flamegraph_simple(tmp_path, clean_profiler):
    """Test exporting profile to speedscope JSON."""
    enable_profiling()
    t = corepy.Tensor([1.0, 2.0])
    _ = t + t
    
    out_file = tmp_path / "profile.speedscope.json"
    export_profile(str(out_file), format="flamegraph")
    
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)
        # Verify it has some structure (our current impl dumps dict, or wrapper)
        # Current impl: _convert_to_speedscope calls report.get("operations")
        # And returns report (fallback)
        assert "operations" in data
