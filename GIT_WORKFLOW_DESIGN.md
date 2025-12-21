# Git-based Development Workflow Design

This document outlines a robust Git-based workflow that allows development on a separate branch without affecting the main branch, and ensures GitHub Actions run and pass before anything reaches main.

## 1. Step-by-Step Git Commands for Developers

This workflow ensures that all development occurs on isolated branches, preventing direct modifications to the `main` branch.

### 1.1 Start a New Feature or Fix
Always start by ensuring your local `main` is up to date, then create a new branch.

```bash
# Switch to main and pull the latest changes
git checkout main
git pull origin main

# Create a new branch for your feature or fix
# Naming convention: feature/description or fix/issue-description
git checkout -b feature/new-login-page
```

### 1.2 Develop and Commit
Make your changes locally. Commit often with clear messages.

```bash
# Show status of changed files
git status

# Stage specific files
git add <file_path>

# Commit changes
git commit -m "feat: implement login form validation"
```

### 1.3 Push and Sync
Push your branch to the remote repository. If you've been working for a while, it's good practice to rebase or merge `main` into your branch to resolve conflicts early.

```bash
# Push the branch to remote
git push -u origin feature/new-login-page
```

### 1.4 Open a Pull Request (PR)
1. Go to the repository on GitHub.
2. You will see a prompt to "Compare & pull request" for your recently pushed branch.
3. Click it and fill in the details.
   - **Base branch**: `main`
   - **Compare branch**: `feature/new-login-page`
4. Assign reviewers and link relevant issues.

### 1.5 Update PR based on specific feedback
If changes are requested or CI fails:

```bash
# Make necessary fixes
# ... edit files ...

git add .
git commit -m "fix: address review comments"
git push origin feature/new-login-page
```
*Note: The Pull Request will automatically update and re-run CI checks.*

---

## 2. GitHub Actions Configuration

Ensure your workflow files (e.g., `.github/workflows/ci.yml`) are configured to trigger on Pull Requests targeting `main`.

```yaml
name: CI

on:
  # Trigger checks on Pull Requests targeting the main branch
  pull_request:
    branches: [ main ]

  # Optional: Also run on main commits (post-merge) to update caches/deploy
  push:
    branches: [ main ]

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: pip install -e ".[dev]"
        
      - name: Run Tests
        run: pytest
```

---

## 3. Branch Protection Rules

You must configure **Branch Protection Rules** on the `main` branch to enforce this workflow.

### Setup Instructions
1. Navigate to **Settings** > **Branches** in your GitHub repository.
2. Click **Add branch protection rule**.
3. **Branch name pattern**: `main`

### Required Settings
| Setting | Description |
| :--- | :--- |
| **Require a pull request before merging** | Ensures no one can push directly to `main`. All changes must come via PR. |
| &nbsp;&nbsp;└─ *Require approvals* | Set to at least **1**. Ensures another pair of eyes reviews the code. |
| **Require status checks to pass before merging** | Critical for CI. The merge button will be disabled until tests pass. |
| &nbsp;&nbsp;└─ *Status checks found in the last week* | Search for and select your CI job name (e.g., `test` or `Run Tests`). |
| **Require branches to be up to date before merging** | Ensures the PR is tested against the *latest* `main` code, preventing regression from semantic conflicts. |
| **Do not allow bypassing the above settings** | Enforces these rules for administrators as well (optional but recommended). |

---

## 4. Best Practices for Production

1. **Semantic Commit Messages**: Use prefixes like `feat:`, `fix:`, `docs:` to make history readable (e.g., [Conventional Commits](https://www.conventionalcommits.org/)).
2. **Squash Merging**: Configure the repository to "Allow squash merging". This keeps the `main` history clean by combining all PR commits into a single commit upon merge.
3. **Draft PRs**: Open PRs as "Draft" if they are work-in-progress. This signals to reviewers that it's not ready for full review yet but runs CI to give you early feedback.
4. **Code Owners**: Use a `CODEOWNERS` file to automatically request reviews from specific team members based on the files changed.
5. **Keep PRs Small**: Smaller PRs are easier to review and less likely to contain hidden bugs.
