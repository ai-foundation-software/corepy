#!/usr/bin/env python3
"""
Test script to validate corepy 0.2.4 released package APIs.
This will help us determine which examples work for usage.md documentation.
"""

print("=" * 60)
print("Testing Corepy 0.2.4 Released Package")
print("=" * 60)

# Test 1: Basic imports
print("\n[Test 1] Basic Imports")
try:
    import corepy as cp

    print(f"✓ Imported corepy version: {cp.__version__}")
except ImportError as e:
    print(f"✗ Failed to import corepy: {e}")
    exit(1)

# Test 2: Check available attributes
print("\n[Test 2] Available Top-Level Attributes")
available = [attr for attr in dir(cp) if not attr.startswith("_")]
print(f"Available: {', '.join(available)}")

# Test 3: Array creation (NumPy-style)
print("\n[Test 3] Array Creation")
try:
    # Primary API
    a = cp.array([1.0, 2.0, 3.0])
    print(f"✓ Created array: {a}")

    # Factory functions
    z = cp.zeros((2, 2))
    print(f"✓ Created zeros: {z}")

    # Backward compatibility
    from corepy import Tensor

    t = Tensor([1, 2, 3])
    print(f"✓ Created legacy Tensor: {t}")
except Exception as e:
    print(f"✗ Array creation failed: {e}")

# Test 4: Array operations
print("\n[Test 4] Array Operations")
try:
    b = cp.array([4.0, 5.0, 6.0])
    c = a + b
    print(f"✓ Addition: {a} + {b} = {c}")

    print(
        f"✓ Properties: shape={c.shape}, dtype={c.dtype}, ndim={c.ndim}, size={c.size}"
    )
except Exception as e:
    print(f"✗ Array operations failed: {e}")

# Test 5: Data module
print("\n[Test 5] Data Module")
try:
    from corepy import data

    print(
        f"✓ Data module available: {', '.join([x for x in dir(data) if not x.startswith('_')])}"
    )
except Exception as e:
    print(f"✗ Data module failed: {e}")

# Test 6: Schema module
print("\n[Test 6] Schema Module")
try:
    from corepy import schema

    print(
        f"✓ Schema module available: {', '.join([x for x in dir(schema) if not x.startswith('_')])}"
    )
except Exception as e:
    print(f"✗ Schema module failed: {e}")

# Test 7: Backend module
print("\n[Test 7] Backend Module")
try:
    from corepy import backend

    print(
        f"✓ Backend module available: {', '.join([x for x in dir(backend) if not x.startswith('_')])}"
    )

    # Test reference backend
    from corepy.backend import ReferenceBackend

    # Usage depends on backend impl, but verifying import is good
    print("✓ ReferenceBackend available")
except Exception as e:
    print(f"✗ Backend module failed: {e}")

# Test 8: C++ extension
print("\n[Test 8] C++ Extension")
try:
    result = cp.add_one(5)
    print(f"✓ C++ add_one(5) = {result}")
except Exception as e:
    print(f"✗ C++ extension failed: {e}")

# Test 9: Runtime module
print("\n[Test 9] Runtime Module")
try:
    from corepy import runtime

    print(
        f"✓ Runtime module available: {', '.join([x for x in dir(runtime) if not x.startswith('_')])}"
    )
except Exception as e:
    print(f"✗ Runtime module failed: {e}")

# Test 10: Type system
print("\n[Test 10] Type System")
try:
    # Try to access type definitions
    if hasattr(cp, "Float32"):
        print("✓ cp.Float32 available")
    else:
        print("⚠ cp.Float32 not found at top level")

    # Check in schema
    from corepy.schema import base

    types_available = [x for x in dir(base) if "Float" in x or "Int" in x]
    print(f"✓ Types in schema.base: {', '.join(types_available)}")
except Exception as e:
    print(f"✗ Type system check failed: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
