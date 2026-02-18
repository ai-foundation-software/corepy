import glob
import os
import re
import subprocess
import sys


def run_python_code(code, description):
    """Runs a snippet of Python code."""
    print(f"Testing: {description}...", end=" ", flush=True)
    try:
        # Add project root to PYTHONPATH
        env = os.environ.copy()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")

        # Run in a subprocess to isolate environments and capture output
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        print("✅ PASS")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAIL")
        print(f"  Error: {e.stderr}")
        print(f"  Output: {e.stdout}")
        return False


def run_python_file(filepath):
    """Runs a Python file."""
    print(f"Testing file: {filepath}...", end=" ", flush=True)
    try:
        # Add project root to PYTHONPATH
        env = os.environ.copy()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")

        # Run in a subprocess
        process = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        print("✅ PASS")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAIL")
        print(f"  Error: {e.stderr}")
        print(f"  Output: {e.stdout}")
        return False


def extract_code_blocks(markdown_path):
    """Extracts Python code blocks from a markdown file."""
    with open(markdown_path, "r") as f:
        content = f.read()

    # Regex to find python code blocks
    # Matches ```python ... ```
    pattern = r"```python\s+(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(base_dir, "README.md")
    examples_dir = os.path.join(base_dir, "examples")
    tutorials_dir = os.path.join(base_dir, "tutorials")

    failures = []

    # 1. Test README.md code blocks
    print(f"\n--- Checking {readme_path} ---")
    if os.path.exists(readme_path):
        code_blocks = extract_code_blocks(readme_path)
        for i, code in enumerate(code_blocks):
            # Skip blocks that are just `pip install` or shell commands if labeled python by mistake,
            # but regex specifically looks for ```python.

            # Heuristic: skip if it doesn't import anything and looks like a fragment
            if "import corepy" not in code and "import" not in code:
                print(f"Skipping block {i + 1} (no imports detected)...")
                continue

            if not run_python_code(code, f"README Block {i + 1}"):
                failures.append(f"README Block {i + 1}")
    else:
        print(f"Warning: {readme_path} not found.")

    # 2. Test examples/
    print(f"\n--- Checking {examples_dir} ---")
    if os.path.exists(examples_dir):
        for filepath in glob.glob(os.path.join(examples_dir, "*.py")):
            if not run_python_file(filepath):
                failures.append(f"File: {os.path.basename(filepath)}")
    else:
        print(f"Warning: {examples_dir} not found.")

    # 3. Test tutorials/
    print(f"\n--- Checking {tutorials_dir} ---")
    if os.path.exists(tutorials_dir):
        # Recursive search for .py files
        for root, dirs, files in os.walk(tutorials_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    if not run_python_file(filepath):
                        failures.append(
                            f"Tutorial: {os.path.relpath(filepath, base_dir)}"
                        )

    # Summary
    print("\n--- Summary ---")
    if failures:
        print(f"❌ Found {len(failures)} failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✅ All documentation examples and scripts passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
