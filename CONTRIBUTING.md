## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Running Linters and Formatters](#running-linters-and-formatters)
- [Running Tests](#running-tests)
- [Coding Style](#coding-style)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Submitting Changes (Pull Requests)](#submitting-changes-pull-requests)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a Code of Conduct (You might want to add a separate `CODE_OF_CONDUCT.md` file later if the project grows, perhaps based on the Contributor Covenant). Please be respectful and constructive in all interactions.

## Setting Up Your Development Environment

1. **Clone the repository:**

   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **(Optional) Set Python Version with `pyenv`:**
   If you use `pyenv` to manage Python versions, ensure you have the required version installed (e.g., specified in `pyproject.toml` or `README.md`) and set it for this project directory:

   ```bash
   # Example: If the project requires Python 3.9.x
   pyenv install 3.9.18 # If not already installed
   pyenv local 3.9.18   # Set the version for this directory
   ```

3. **Create a virtual environment:**
   It's highly recommended to use a virtual environment (like `venv`) to manage dependencies. Using the Python version set in the previous step (or your system's default if not using `pyenv`):

   ```bash
   python -m venv .venv
   # Activate the environment
   # On Windows (Git Bash or cmd.exe)
   # .venv\Scripts\activate
   # On macOS/Linux (bash/zsh)
   source .venv/bin/activate
   ```

   _Note: Some tools like `pyenv-virtualenv` can combine steps 2 and 3._

4. **Install dependencies:**
   Install the required Python packages into the active virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

5. **Install development tools:**
   Install tools used for formatting, linting, testing, and pre-commit hooks.

   ```bash
   pip install ruff mypy pre-commit pytest pytest-cov # Add other dev tools if needed (e.g., black if not using ruff format)
   ```

6. **Install pre-commit hooks:**
   This step configures Git to run checks automatically before each commit, based on the `.pre-commit-config.yaml` file.

   ```bash
   pre-commit install
   ```

## Running Linters and Formatters

We use `ruff` for linting and formatting, and `mypy` for type checking. These are configured in `pyproject.toml`.

**Automatic Checks (Recommended):**

If you followed step 6 in the setup (`pre-commit install`), the necessary checks (formatting, linting, type checking) will run automatically on staged files whenever you run `git commit`. If any check fails, the commit will be aborted. Some tools (like `ruff format` and `ruff --fix`) may automatically modify files to fix issues; simply `git add` the modified files and try committing again.

**Manual Checks:**

You can also run the checks manually:

- **Run all pre-commit hooks on staged files:**

  ```bash
  pre-commit run
  ```

- **Run all pre-commit hooks on all files:**

  ```bash
  pre-commit run --all-files
  ```

- **Check for linting errors with Ruff:**

  ```bash
  ruff check .
  ```

- **Format code with Ruff:**

  ```bash
  ruff format .
  ```

- **Run type checking with Mypy:**

  ```bash
  mypy src tests # Adjust paths as needed, ensure it uses pyproject.toml config
  ```

## Running Tests

We use `pytest` for running tests. Ensure all tests pass before submitting changes.

- **Run all tests:**

  ```bash
  pytest
  ```

- **Run tests with coverage report:**
  (This uses the configuration in `pyproject.toml`)

  ```bash
  pytest --cov
  ```

## Coding Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
- Use `ruff format` (via pre-commit or manually) for automatic code formatting.
- Use `ruff check` (via pre-commit or manually) for linting. Adhere to the rules defined in `pyproject.toml`.
- Write clear and concise comments where necessary, explaining the _why_ behind complex logic.
- Use type hints for function signatures and variables where appropriate, and ensure `mypy` passes.

## Commit Message Guidelines

Please follow conventional commit guidelines for clear and automated changelog generation (optional but good practice):

- **Format:** `<type>[optional scope]: <description>`
- **Examples:**
  - `feat: Add KMeans clustering for stage labeling`
  - `fix: Correct calculation for time-to-next-stage`
  - `docs: Update README with setup instructions`
  - `style: Format code using ruff format`
  - `refactor: Improve data loading efficiency`
  - `test: Add unit tests for risk score calculation`
  - `chore: Configure pre-commit hooks`

Common types: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`.

## Submitting Changes (Pull Requests)

(This section is more relevant if collaborating, but good to define the process)

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b <type>/<short-description>` (e.g., `feat/add-xgboost-model`, `fix/stage-label-off-by-one`).
3. Make your changes. Ensure pre-commit checks pass when you commit. Ensure all tests pass (`pytest`).
4. Commit your changes using clear commit messages.
5. Push your branch to your fork: `git push origin <branch-name>`.
6. Open a Pull Request (PR) against the `main` (or `develop`) branch of the original repository.
7. Provide a clear description of the changes in the PR. Link to any relevant issues.
8. Respond to any feedback or requested changes during the review process.

## Reporting Bugs

If you find a bug, please open an issue on the project's issue tracker (e.g., GitHub Issues). Include:

- A clear and descriptive title.
- Steps to reproduce the bug.
- What you expected to happen.
- What actually happened (including error messages or tracebacks).
- Your environment details (OS, Python version, library versions).

## Suggesting Enhancements

If you have an idea for an improvement or a new feature:

1. Check the issue tracker to see if a similar suggestion already exists.
2. If not, open a new issue.
3. Provide a clear description of the enhancement and why it would be beneficial.
4. Explain the proposed solution or workflow if possible.

Thank you for contributing!
