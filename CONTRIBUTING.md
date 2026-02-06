# Contributing to Honey-Prompt Detector

Thank you for your interest in contributing to the Honey-Prompt Detector! We welcome contributions from the community to help improve this project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Making Changes](#making-changes)
4. [Testing](#testing)
5. [Code Style](#code-style)
6. [Submitting Changes](#submitting-changes)
7. [Reporting Issues](#reporting-issues)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

Before starting any major work, **please reach out to the maintainers first** to discuss your ideas. This helps:
- Avoid duplicate work
- Ensure your contribution aligns with the project's goals
- Get early feedback on your approach

You can reach out by:
- Opening an issue to discuss your proposed changes
- Commenting on existing issues you'd like to work on

---

## Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/Yaima/honey-prompt-detector.git
cd honey-prompt-detector
```

### 2. Set Up Your Development Environment

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key and other settings
```

### 4. Install Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

---

## Making Changes

### Branch Naming Convention

Create a descriptive branch name:
- `feature/add-new-detector` for new features
- `fix/bug-in-orchestrator` for bug fixes
- `docs/update-readme` for documentation changes
- `refactor/improve-config` for refactoring

```bash
git checkout -b feature/your-feature-name
```

### Code Organization

- **Core Logic**: Place in `src/honey_prompt_detector/core/`
- **Agents**: Place in `src/honey_prompt_detector/agents/`
- **Utilities**: Place in `src/honey_prompt_detector/utils/`
- **Monitoring**: Place in `src/honey_prompt_detector/monitoring/`
- **Tests**: Place in `test/`

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/honey_prompt_detector --cov-report=term-missing

# Run specific test file
pytest test/test_detector.py

# Run specific test
pytest test/test_detector.py::test_exact_match
```

### Writing Tests

- Write tests for all new features
- Ensure tests are async-compatible when needed
- Aim for high code coverage (>80%)
- Include both unit tests and integration tests

Example test structure:

```python
import pytest
from src.honey_prompt_detector.core.detector import Detector

class TestDetector:
    @pytest.fixture
    async def detector(self):
        # Setup code
        return Detector(...)

    async def test_exact_match(self, detector):
        # Test code
        result = detector.analyze_text("test input", ...)
        assert result['matched'] is True
```

### Running Experiments

```bash
# Run the comprehensive experiment suite
python -m src.honey_prompt_detector.main --run-experiments
```

---

## Code Style

We follow PEP 8 style guidelines with some modifications:

### Formatting

- **Line Length**: Maximum 120 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized with `isort`
- **Code Formatting**: Formatted with `black`

### Running Code Quality Tools

```bash
# Format code with black
black src/ test/

# Sort imports with isort
isort src/ test/

# Check code style with flake8
flake8 src/ test/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks

If you installed pre-commit hooks, these checks will run automatically before each commit.

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.

    Detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When parameter is invalid
    """
    pass
```

---

## Submitting Changes

### 1. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new detection algorithm for obfuscated tokens"
```

#### Commit Message Convention

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `style:` Code style changes (formatting, etc.)
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template with:
   - Description of changes
   - Related issues (if any)
   - Testing performed
   - Screenshots (if applicable)

### 4. Code Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

---

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Python version
   - Operating system
   - Relevant package versions
6. **Logs/Screenshots**: Any relevant logs or screenshots

### Feature Requests

When requesting features, please include:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your idea for how to solve it
3. **Alternatives**: Any alternative solutions you've considered
4. **Additional Context**: Any other relevant information

---

## Community Guidelines

### Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Be Respectful

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what's best for the community
- Show empathy towards others

### Security Issues

If you discover a security vulnerability, please email us at yaimamvaldivia@gmail.com instead of opening a public issue. See our [Security Policy](SECURITY.md) for more details.

---

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Join our community discussions

---

Thank you for contributing to Honey-Prompt Detector!
