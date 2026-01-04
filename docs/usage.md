# 📖 Usage Guide

Corepy is designed to be familiar if you have used tools like NumPy or Pandas, but with added safety and speed.

## 🧠 Core Concepts

1.  **Correctness First**: Corepy will error out rather than give you a wrong answer.
2.  **Lazy Execution**: Sometimes, when you tell Corepy to do math, it doesn't do it generic immediately. It creates a plan (a "graph") and runs it efficiently when you ask for the result.
3.  **Device Aware**: You can move data between CPU and GPU easily.

---

## 🌟 Example 1: Basic Math (The "Hello World")

Let's create two request processing tensors and add them.

```python
import corepy as cp

# Create two tensors (arrays of numbers)
# We specify the type 'Float32' to be precise.
a = cp.tensor([1.0, 2.0, 3.0], dtype=cp.Float32)
b = cp.tensor([4.0, 5.0, 6.0], dtype=cp.Float32)

# Add them together
# This happens efficiently using C++ optimized code.
c = a + b

print(c)
# Output: [5.0, 7.0, 9.0]
```

---

## 🖼️ Example 2: AI Preprocessing Pipeline

A common use for Corepy is preparing images for AI training. We need to load them, resize them, and normalize the pixel values.

```python
import corepy as cp

def preprocess_images(folder_path):
    # 1. Find all images
    # This just creates a list, it doesn't load them yet.
    files = cp.io.glob(f"{folder_path}/*.jpg")
    
    # 2. Load the images in parallel
    # Corepy uses its Rust scheduler to load many files at once
    # without blocking your main program.
    images_batch = cp.io.read_images(files)
    
    # 3. Normalize pixel values
    # Standard formula: (pixel - mean) / std_dev
    # This creates a "computation graph" - it's a plan of what to do.
    normalized = (images_batch - 0.485) / 0.229
    
    # 4. Resize
    # Uses fast C++ SIMD instructions.
    resized = cp.vision.resize(normalized, (224, 224))
    
    # 5. Execute!
    # Up until now, we were just building the plan.
    # .compute() runs the plan on the best available hardware.
    return resized.compute(device="auto")

# Run the pipeline
batch = preprocess_images("./my_data")
print(f"Processed batch shape: {batch.shape}")
```

---

## 📈 Example 3: High-Frequency Data (Safe Concurrency)

If you are processing financial data, you can't afford race conditions (where two parts of code fight over memory). Corepy handles this safely.

```python
import corepy.data as cpd
import corepy as cp

# Read a large file using 'Arrow' format (very fast)
df = cpd.read_ipc("trade_data.arrow")

# Calculate metrics
# We group by symbol and calculate a rolling average.
# The heavy lifting is distributed across CPU cores automatically.
stats = (
    df.select("symbol", "price")
    .group_by("symbol")
    .rolling(window="1s")  # 1 second window
    .agg(
        avg_price=cp.mean("price")
    )
)

print(stats.head())
```

---

## 🚫 What NOT to do (Anti-Patterns)

**1. Don't loop in Python if you can help it.**
```python
# ❌ SLOW: Python loop
result = []
for x in data:
    result.append(x * 2)

# ✅ FAST: Corepy vectorized operation
result = data * 2
```

**2. Don't worry about thread pools.**
Corepy has a built-in scheduler. You don't need to use `threading` or `multiprocessing`. Just write your code, and let Corepy handle the parallelism.
