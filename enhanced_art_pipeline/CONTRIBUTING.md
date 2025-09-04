# Contributing Guide

## Development Setup
1. Ensure Python 3.10 or newer is installed.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Coding Style & Commit Rules
- Format code with `black` and check lint with `flake8`.
- Follow PEP 8 and document public functions.
- Use imperative commit messages (e.g., `Add feature` not `Added feature`).
- Keep commits focused on a single topic.

## Running Tests
- Execute the test suite before pushing:
  ```bash
  pytest
  ```

## Pull Request Process
1. Create a branch and make your changes.
2. Run formatters, linters, and tests.
3. Submit a PR using the template and reference related issues.
4. Ensure all checks pass before requesting review.
