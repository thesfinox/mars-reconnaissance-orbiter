# Contributing Guide

We welcome contributions to the HSI Mars project! This guide will help you get started.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Issues](#reporting-issues)
- [Community Guidelines](#community-guidelines)
- [Getting Help](#getting-help)

## Ways to Contribute

There are many ways to contribute to this project:

- 🐛 **Report bugs** - Help us identify and fix issues
- 💡 **Suggest new features** - Share ideas for improvements
- 📝 **Improve documentation** - Make our docs clearer and more comprehensive
- 🧪 **Add tests** - Increase code coverage and reliability
- 💻 **Submit code improvements** - Fix bugs or implement features
- 📊 **Share example analyses** - Contribute notebooks or examples

## Getting Started

### 1. Fork the Repository

Visit the [GitHub repository](https://github.com/thesfinox/mars-reconnaissance-orbiter) and click the "Fork" button in the top right.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/mars-reconnaissance-orbiter.git
cd mars-reconnaissance-orbiter
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Set Up Development Environment

#### Using `uv` (Recommended)

```bash
# Install all development dependencies
uv sync --group dev --group test --group docs

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

#### Using `pip`

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS

# Install in editable mode with all dependencies
pip install -e ".[dev,test,docs]"
```

### 4. Install Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

This ensures code quality checks run automatically before each commit.

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:

- `feature/add-new-visualization` - for new features
- `bugfix/fix-annotation-padding` - for bug fixes
- `docs/improve-installation-guide` - for documentation
- `refactor/simplify-data-loading` - for code refactoring

### 2. Make Your Changes

Follow these guidelines:

- ✅ Write clear, descriptive commit messages
- ✅ Add docstrings to all functions and classes
- ✅ Follow PEP 8 style guidelines
- ✅ Add type hints where appropriate
- ✅ Keep changes focused and atomic
- ✅ Update documentation as needed

### 3. Write Tests

Add tests for new functionality in the `tests/` directory:

```python
# In tests/test_new_feature.py
import pytest
from hsimars import HSIMars


def test_new_feature_produces_expected_output():
    """Verify that new feature works correctly."""
    # Arrange
    hsi = HSIMars(hdr_path="data/test.hdr")

    # Act
    result = hsi.new_method()

    # Assert
    assert result is not None
    assert result.shape == (100, 100)
```

### 4. Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest tests/ --cov=src/hsimars --cov-report=html

# Run specific test file
uv run pytest tests/test_new_feature.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "test_annotation"
```

### 5. Check Code Quality

```bash
# Run linter checks
uv run ruff check src/ tests/

# Check code formatting
uv run ruff format --check src/ tests/

# Auto-fix issues (where possible)
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

### 6. Update Documentation

Update documentation as needed:

- Add NumPy-style docstrings to new functions/classes
- Update relevant `.rst` files in `docs/source/`
- Add examples if introducing new functionality
- Update README.md if needed

Build documentation locally to verify:

```bash
cd docs
uv run make html
# Open docs/build/html/index.html in a browser
```

### 7. Commit Your Changes

```bash
git add .
git commit -m "Add feature: descriptive message"
```

**Writing Good Commit Messages:**

- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Capitalize the first letter
- No period at the end of the summary
- Add detailed description after a blank line if needed

**Examples:**

```
Add spectral angle mapper method

Implement SAM algorithm for spectral similarity comparison.
Includes unit tests and documentation.
```

```
Fix annotation padding calculation

Fixes issue where annotations were clipped at image edges.
Resolves #123.
```

### 8. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then:

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Select your branch and describe your changes
4. Submit the pull request

## Code Style Guidelines

### Python Style

Follow **PEP 8** with these specifics:

- **Line length**: 88 characters (Black/ruff default)
- **Indentation**: 4 spaces (no tabs)
- **Quote style**: Double quotes for strings
- **Imports**: Organized and sorted
- **Type hints**: Use `from __future__ import annotations`

**Example:**

```python
"""Module for data processing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from spectral import open_image


def process_data(
    data: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Process data with given threshold.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    threshold : float, optional
        Processing threshold. Default is 0.5.

    Returns
    -------
    np.ndarray
        Processed data array.

    Examples
    --------
    >>> data = np.array([0.3, 0.7, 0.9])
    >>> process_data(data, threshold=0.5)
    array([0.7, 0.9])
    """
    return data[data > threshold]
```

### Documentation Style

Use **NumPy-style docstrings** for all public functions and classes:

```python
def example_function(param1: str, param2: int | None = None) -> bool:
    """
    Brief description of the function.

    Detailed description if needed. Can span multiple
    paragraphs and include examples.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2. Default is None.

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is invalid or empty.
    TypeError
        When param2 is not an integer.

    Examples
    --------
    >>> result = example_function("test", 42)
    >>> print(result)
    True

    Notes
    -----
    Additional information about implementation or usage.
    Can include references to papers or algorithms.

    See Also
    --------
    related_function : Related functionality.
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    return True
```

### Import Organization

Organize imports in this order:

1. Standard library imports
2. Related third-party imports
3. Local application imports

**Example:**

```python
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import spectral as sp
from matplotlib import pyplot as plt

from hsimars.utils import load_data
```

## Testing Guidelines

### Test Organization

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain intent

**Examples of good test names:**

```python
def test_get_img_returns_correct_dimensions_for_test_dataset():
    """Verify that get_img returns expected dimensions."""
    pass

def test_annotation_raises_value_error_for_negative_coordinates():
    """Ensure negative coordinates raise ValueError."""
    pass
```

### Test Structure

Follow the **Arrange-Act-Assert (AAA)** pattern:

```python
def test_feature():
    """Test description."""
    # Arrange: Set up test conditions
    hsi = HSIMars(hdr_path="test.hdr")
    expected_shape = (100, 100, 50)

    # Act: Execute the functionality
    result = hsi.get_img()

    # Assert: Verify the outcome
    assert result.shape == expected_shape
    assert result.dtype == np.float32
```

### Using Fixtures

Use pytest fixtures from `conftest.py` for common test data:

```python
def test_with_fixture(stub_hsi):
    """Test using stub HSI data fixture."""
    # stub_hsi is automatically provided by conftest.py
    result = stub_hsi.get_img()
    assert result is not None
```

### Coverage Goals

Aim for high test coverage:

- ✅ All public methods should have tests
- ✅ Critical code paths must be tested
- ✅ Edge cases should be covered
- ✅ Non-regression tests for bug fixes
- ✅ Target: >80% code coverage

Check coverage:

```bash
uv run pytest tests/ --cov=src/hsimars --cov-report=term-missing
```

## Pull Request Guidelines

### Before Submitting

Ensure your PR:

- ✅ Passes all tests (`uv run pytest tests/`)
- ✅ Passes linting checks (`uv run ruff check src/ tests/`)
- ✅ Maintains or improves code coverage
- ✅ Follows code style guidelines
- ✅ Includes documentation updates
- ✅ Has a clear, descriptive title
- ✅ Includes a detailed description

### PR Description Template

Use this template for your pull request:

```markdown
## Description

Brief description of what this PR does and why.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Related Issues

Fixes #(issue number)
Relates to #(issue number)

## Testing

Describe the tests you ran and their results:

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed (describe)

## Screenshots (if applicable)

Add screenshots for visual changes.

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Documentation updated (docstrings, README, etc.)
- [ ] Tests added/updated and all tests pass
- [ ] No new warnings introduced
- [ ] Changes are backwards compatible (or breaking changes are documented)
```

### Code Review Process

After submission:

1. **Automated checks run** - GitHub Actions runs tests, linting, docs build
2. **Maintainer reviews** - Project maintainer reviews your code
3. **Address feedback** - Make requested changes and push to your branch
4. **Approval and merge** - Once approved, PR is merged into main branch

**Tips for getting PRs merged faster:**

- Keep PRs focused and reasonably sized
- Respond promptly to review comments
- Ensure CI checks pass before requesting review
- Write clear PR descriptions

## Reporting Issues

### Bug Reports

When reporting a bug, include:

1. **Environment information**:
   - Python version
   - Package version (`pip show hsi-mars`)
   - Operating system

2. **Description**:
   - What you expected to happen
   - What actually happened
   - Error messages/tracebacks (full output)

3. **Minimal reproducible example**:

```python
from hsimars import HSIMars

# Minimal code that reproduces the issue
hsi = HSIMars(hdr_path="path/to/file.hdr")
result = hsi.problematic_method()  # This causes the error
```

4. **Steps to reproduce**:
   - Step-by-step instructions
   - Sample data if needed

### Feature Requests

When requesting a feature, include:

1. **Clear description** - What functionality do you want?
2. **Use case and motivation** - Why is this needed?
3. **Proposed implementation** - How might it work? (optional)
4. **Examples** - Similar features in other tools
5. **Alternatives** - Other ways to achieve the goal

### Issue Labels

Use appropriate labels:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `question` - Questions about usage
- `help wanted` - Extra attention needed
- `good first issue` - Good for newcomers

## Community Guidelines

### Be Respectful

- ✅ Use welcoming and inclusive language
- ✅ Respect differing viewpoints and experiences
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what's best for the community
- ✅ Show empathy towards other contributors

### Communication

- Be clear and concise in your communication
- Provide context for your contributions
- Ask questions when unsure - no question is too small
- Help others when you can
- Be patient with new contributors

### Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Report unacceptable behavior to <riccardo.finotello@cea.fr>.

## Getting Help

If you need help contributing:

- 📖 **Read the documentation** - Check [docs](https://hsi-mars.readthedocs.io/)
- 🔍 **Search existing issues** - Your question may already be answered
- 💬 **Ask in GitHub Discussions** - Start a discussion for questions
- 📧 **Contact the maintainer** - <riccardo.finotello@cea.fr>

## Development Tips

### Local Testing

Test your changes locally before pushing:

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=src/hsimars --cov-report=html
# Open htmlcov/index.html to view coverage report

# Run only fast tests (if marked)
uv run pytest tests/ -m "not slow"

# Run tests in parallel (faster)
uv run pytest tests/ -n auto
```

### Building Documentation

Build and preview documentation:

```bash
# Build HTML documentation
cd docs
uv run make html

# Clean previous builds
uv run make clean

# Build and check for warnings
uv run sphinx-build -W -b html source build/html

# Serve documentation locally (Python 3)
cd build/html
python -m http.server 8000
# Open http://localhost:8000 in browser
```

### Debugging Tests

Debug failing tests:

```bash
# Run with verbose output
uv run pytest tests/test_file.py -v

# Run specific test
uv run pytest tests/test_file.py::test_function_name -v

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Show print statements
uv run pytest tests/ -s

# Show local variables on failure
uv run pytest tests/ -l
```

### Working with Notebooks

When contributing Jupyter notebooks:

1. **Clear outputs before committing** (optional, depending on notebook):

   ```bash
   jupyter nbconvert --clear-output --inplace notebook.ipynb
   ```

2. **Run notebooks to ensure they work**:

   ```bash
   jupyter nbconvert --to notebook --execute notebook.ipynb
   ```

3. **Keep notebooks focused** - One topic per notebook
4. **Add markdown cells** - Explain what the code does
5. **Use small datasets** - For examples and tutorials

## Recognition

All contributors are acknowledged in:

- 🌟 **GitHub contributors list** - Automatically tracked
- 📋 **Release notes** - Mentioned in changelog
- 📖 **Project documentation** - Listed as contributors

Thank you for contributing to HSI Mars! 🎉

---

**Questions?** Feel free to reach out:

- **Email**: <riccardo.finotello@cea.fr>
- **GitHub Issues**: [Open an issue](https://github.com/thesfinox/mars-reconnaissance-orbiter/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/thesfinox/mars-reconnaissance-orbiter/discussions)
