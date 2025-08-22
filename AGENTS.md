# AGENTS Guidelines

## Directory Structure
- `art_outlines/`: Core models, configuration, and scripts.
- `docs/`: Design references and project documentation.
- `swiftsketch/`: SwiftSketch resources.

## Build and Test
- Use **Python 3.10+**.
- Install dependencies: `pip install -r requirements.txt`.
- Run tests: `pytest`.
- Format code with `black` and lint with `flake8`.

## Code Style and PR Rules
- Follow PEP 8 and keep functions small and well‑documented.
- Run formatters and linters before committing.
- Write descriptive commit messages in the imperative mood.
- Every pull request must reference related issues and include test results.

## Warnings
- Do not commit large binary files or secrets.
- Be cautious when modifying training data or configs; consult maintainers if unsure.
