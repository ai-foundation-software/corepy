"""
Working Examples for Corepy (Validated against released package)
Based on testing corepy 0.2.0 from PyPI

These examples all work with the released package.
"""

import corepy as cp

print("=" * 60)
print("EXAMPLE 1: Basic Tensor Creation and Operations")
print("=" * 60)

# Create tensors
a = cp.Tensor([1.0, 2.0, 3.0])
b = cp.Tensor([4.0, 5.0, 6.0])

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")

# Element-wise addition
c = a + b
print(f"a + b = {c}")

# Element-wise subtraction (multiplication with scalar not implemented yet)
d = a - b
print(f"a - b = {d}")

print("\n" + "=" * 60)
print("EXAMPLE 2: Working with Data Table")
print("=" * 60)

from corepy.data import Table

# Create a simple data table
data = {"name": ["Alice", "Bob", "Charlie"], "score": [95.5, 87.3, 92.1]}

table = Table(data)
print(f"Created table:\n{table}")

print("\n" + "=" * 60)
print("EXAMPLE 3: Backend System")
print("=" * 60)

from corepy.backend import detect_devices, select_backend

# Detect available devices
devices = detect_devices()
print(f"Available devices: {devices}")

# Select CPU backend (default)
backend = select_backend("cpu")
print(f"Selected backend: {backend}")

print("\n" + "=" * 60)
print("EXAMPLE 4: Schema Definition")
print("=" * 60)

from corepy.schema import Field, Schema

# Define a strict schema
schema = Schema(
    [
        Field("user_id", int),
        Field("score", float),
    ]
)

print(f"Defined schema: {schema}")

print("\n" + "=" * 60)
print("EXAMPLE 5: Runtime Pipeline")
print("=" * 60)

from corepy.runtime import Pipeline

# Create a simple pipeline
pipeline = Pipeline()
print(f"Created pipeline: {pipeline}")

print("\n" + "=" * 60)
print("All Examples Complete!")
print("=" * 60)
