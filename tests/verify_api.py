import numpy as np

import corepy as cp


def verify_api():
    print("=== API Verification ===")

    a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    a_cp = cp.ndarray(a_np)
    b_cp = cp.ndarray(b_np)

    expected = 32.0  # 1*4 + 2*5 + 3*6 = 4+10+18 = 32

    # 1. Test cp.dot()
    try:
        res_dot = cp.dot(a_cp, b_cp)
        print(
            f"cp.dot(a, b): {res_dot._get_scalar_value()} ... {'✅' if res_dot._get_scalar_value() == expected else '❌'}"
        )
    except Exception as e:
        print(f"cp.dot(a, b): ❌ Error: {e}")

    # 2. Test @ operator
    try:
        res_at = a_cp @ b_cp
        print(
            f"a @ b:        {res_at._get_scalar_value()} ... {'✅' if res_at._get_scalar_value() == expected else '❌'}"
        )
    except Exception as e:
        print(f"a @ b:        ❌ Error: {e}")


if __name__ == "__main__":
    verify_api()
