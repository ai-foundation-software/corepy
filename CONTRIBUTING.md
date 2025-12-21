# Contributing to Corepy

Thank you for your interest in contributing to **corepy**! We welcome contributions from everyone.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/ai-foundation-software/corepy.git
    cd corepy
    ```
3.  **Set up your environment using `uv`**:
    ```bash
    # Create a virtual environment and install dependencies
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[dev,docs]"
    ```

## Development Workflow

1.  Create a new branch for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  Make your changes.
3.  Run tests to ensure everything is working:
    ```bash
    pytest
    ```
4.  Run the linter:
    ```bash
    uv run ruff check .
    ```
5.  Commit your changes using meaningful commit messages.

## Pull Requests

1.  Push your branch to GitHub.
2.  Open a Pull Request against the `main` branch.
3.  Describe your changes and link to any relevant issues.

## Code Style

- We use **Ruff** for linting and formatting.
- Type hints are **mandatory** for all public APIs.
- Docstrings should follow the **Google style**.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
