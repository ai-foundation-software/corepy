# 👋 Contributing to Corepy

First off, thank you for considering contributing to Corepy! Open source lives because of people like you.

Whether you are fixing a typo, adding an example, or writing complex C++ kernels, we welcome your help.

---

## 🌟 Our Philosophy

1.  **Be Kind**: We are a friendly community. Questions are encouraged.
2.  **Correctness First**: It is better to be slow and right than fast and wrong.
3.  **Test Everything**: If it's not tested, it's broken.

---

## 🛠️ Setting Up Your Development Environment

To contribute code, you'll need to build Corepy from source.

1.  **Follow the [Installation Guide](docs/install.md)** to set up your environment (Python, C++, Rust).
2.  **Install Development Dependencies**:
    ```bash
    pip install pytest pytest-cov ruff maturin
    ```
3.  **Run the Tests**:
    Make sure everything is working before you start.
    ```bash
    pytest tests/
    ```

---

## 📝 How to Submit a Change (Pull Request)

1.  **Find an Issue**: Look for issues labeled `good first issue` on GitHub.
2.  **Create a Branch**: `git checkout -b my-new-feature`
3.  **Make your Changes**:
    - If you change Python code, run `ruff check .` to format it.
    - If you change Rust code, run `cargo fmt`.
4.  **Add Tests**:
    - We use a "Sandwich" testing strategy: Python calls Rust, which calls C++.
    - Add a test case in `tests/` that proves your feature works (or that the bug is fixed).
5.  **Submit a PR**: Push your branch and open a Pull Request.

### ✅ PR Checklist
Before you submit, ask yourself:
- [ ] Did I add a test?
- [ ] Did I run the existing tests to make sure I didn't break anything?
- [ ] Is my code clean and readable?
- [ ] Did I update the documentation if needed?

---

## 📂 Project Structure (Where things live)

- `corepy/`: The Python code. This is what users see.
- `rust/`: The Rust runtime. Handles safety and scheduling.
- `csrc/`: The C++ kernels. Handles raw math speed.
- `tests/`: Where we verify correct behavior.
- `docs/`: The documentation you are reading right now.

---

## 🆘 Getting Help

If you get stuck, please open an issue or ask in our discussions. We are happy to mentor new contributors!
