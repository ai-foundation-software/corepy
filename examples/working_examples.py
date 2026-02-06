"""
Working Examples for Corepy (Validated against released package)
Based on testing corepy 0.2.2 from PyPI

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

# Element-wise subtraction
d = a - b
print(f"a - b = {d}")

print("\n" + "=" * 60)
print("EXAMPLE 2: Working with Data Table")
print("=" * 60)

from corepy.data import Table

# Create a simple data table (expects List[Dict[str, Any]])
data = [
    {"name": "Alice", "score": 95.5},
    {"name": "Bob", "score": 87.3},
    {"name": "Charlie", "score": 92.1},
]

table = Table(data)
print(f"Created table:\n{table}")
print(f"Table length: {len(table)}")

print("\n" + "=" * 60)
print("EXAMPLE 3: Backend System")
print("=" * 60)

from corepy.backend import detect_devices
from corepy.backend.types import BackendType, OperationProperties, OperationType

# Detect available devices
devices = detect_devices()
print(f"Available devices: {devices}")

# Select backend properly using the full API
from corepy.backend.selector import select_backend

op_props = OperationProperties(
    element_count=1000,
    shape=(1000,),
    dtype_bytes=4,
)
backend = select_backend(
    OperationType.COMPUTE_VECTOR,
    op_props,
    devices,
)
print(f"Selected backend for 1000 elements: {backend}")

print("\n" + "=" * 60)
print("EXAMPLE 4: Schema Definition")
print("=" * 60)

from corepy.schema import Field, Schema

# Define a strict schema (Field uses string dtype names)
schema = Schema(
    fields=[
        Field(name="user_id", dtype="int"),
        Field(name="score", dtype="float"),
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
