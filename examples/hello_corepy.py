import corepy

def main():
    print(f"Corepy Version: {corepy.__version__}")
    
    # Demonstrate C++ extension
    val = 10
    result = corepy.add_one(val)
    print(f"C++ Kernel: add_one({val}) = {result}")

    # Check Rust extension status
    if hasattr(corepy, "sum_as_string") and corepy.sum_as_string:
        print(f"Rust Kernel: sum_as_string(2, 2) = {corepy.sum_as_string(2, 2)}")
    else:
        print("Rust Kernel: Not fully linked yet (Placeholder active)")

if __name__ == "__main__":
    main()
