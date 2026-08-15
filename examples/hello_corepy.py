import corepy as cp


def main():
    print(f"Corepy Version: {cp.__version__}")
    print("-" * 30)

    # 1. Hardware Awareness
    print("Backend Analysis:")
    analysis = cp.analyse_workload(matrix_size=1024)
    print(analysis)
    print("-" * 30)

    # 2. Modern ndarray Operations
    a = cp.array([10.0, 20.0, 30.0])
    b = cp.array([1.0, 2.0, 3.0])

    print(f"ndarray a: {a}")
    print(f"ndarray b: {b}")

    # Simple addition
    c = a + b
    print(f"a + b = {c}")

    # Scalar multiplication
    d = a * 2.0
    print(f"a * 2.0 = {d}")


if __name__ == "__main__":
    main()
