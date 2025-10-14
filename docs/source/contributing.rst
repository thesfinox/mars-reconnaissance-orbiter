Contributing Guide
==================

We welcome contributions to the HSI Mars project! This guide will help you get started.

Ways to Contribute
------------------

There are many ways to contribute to this project:

* 🐛 Report bugs
* 💡 Suggest new features
* 📝 Improve documentation
* 🧪 Add tests
* 💻 Submit code improvements
* 📊 Share example analyses

Getting Started
---------------

1. Fork the Repository
~~~~~~~~~~~~~~~~~~~~~~~

Visit the `GitHub repository <https://github.com/thesfinox/mars-reconnaissance-orbiter>`_ and click the "Fork" button.

2. Clone Your Fork
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/mars-reconnaissance-orbiter.git
   cd mars-reconnaissance-orbiter

3. Set Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using ``uv`` (recommended):

.. code-block:: bash

   # Install all development dependencies
   uv sync --group dev --group test --group docs

   # Activate the virtual environment
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate  # On Windows

Using ``pip``:

.. code-block:: bash

   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS

   # Install in editable mode with all dependencies
   pip install -e ".[dev,test,docs]"

4. Install Pre-commit Hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pre-commit install

This ensures code quality checks run automatically before each commit.

Development Workflow
--------------------

1. Create a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name

Use descriptive branch names:

* ``feature/add-new-visualization``
* ``bugfix/fix-annotation-padding``
* ``docs/improve-installation-guide``

2. Make Your Changes
~~~~~~~~~~~~~~~~~~~~

Follow these guidelines:

* Write clear, descriptive commit messages
* Add docstrings to all functions and classes
* Follow PEP 8 style guidelines
* Add type hints where appropriate
* Keep changes focused and atomic

3. Write Tests
~~~~~~~~~~~~~~

Add tests for new functionality:

.. code-block:: python

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

4. Run Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with coverage
   pytest tests/ --cov=hsimars --cov-report=html

   # Run specific test file
   pytest tests/test_new_feature.py -v

5. Update Documentation
~~~~~~~~~~~~~~~~~~~~~~~

Update documentation as needed:

* Add docstrings to new functions/classes
* Update relevant ``.rst`` files in ``docs/source/``
* Add examples if introducing new functionality

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   # Open docs/build/html/index.html in a browser

6. Commit Your Changes
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add .
   git commit -m "Add feature: descriptive message"

Write good commit messages:

* Use present tense ("Add feature" not "Added feature")
* Keep first line under 50 characters
* Add detailed description if needed

7. Push and Create Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/your-feature-name

Then visit GitHub and create a Pull Request.

Code Style Guidelines
---------------------

Python Style
~~~~~~~~~~~~

Follow PEP 8 with these specifics:

* **Line length**: 80 characters (enforced by ruff)
* **Indentation**: 4 spaces
* **Quote style**: Double quotes for strings
* **Imports**: Organized and sorted

Example:

.. code-block:: python

   """Module docstring."""

   from __future__ import annotations

   from pathlib import Path
   from typing import NamedTuple

   import numpy as np


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
       """
       return data[data > threshold]

Documentation Style
~~~~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def example_function(param1, param2):
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
           When param1 is invalid.

       Examples
       --------
       >>> result = example_function("test", 42)
       >>> print(result)
       True

       Notes
       -----
       Additional information about implementation or usage.
       """
       pass

Testing Guidelines
------------------

Test Organization
~~~~~~~~~~~~~~~~~

* Place tests in ``tests/`` directory
* Name test files ``test_*.py``
* Name test functions ``test_*``
* Use descriptive test names that explain intent

Example:

.. code-block:: python

   def test_get_img_returns_correct_dimensions_for_test_dataset():
       """Verify that get_img returns expected dimensions."""
       # Test implementation

Test Structure
~~~~~~~~~~~~~~

Follow the Arrange-Act-Assert pattern:

.. code-block:: python

   def test_feature():
       """Test description."""
       # Arrange: Set up test conditions
       hsi = HSIMars(hdr_path="test.hdr")
       expected = (100, 100, 50)

       # Act: Execute the functionality
       result = hsi.get_img()

       # Assert: Verify the outcome
       assert result.shape == expected

Coverage
~~~~~~~~

Aim for high test coverage:

* All public methods should have tests
* Critical code paths must be tested
* Edge cases should be covered
* Non-regression tests for bug fixes

Pull Request Guidelines
-----------------------

Before Submitting
~~~~~~~~~~~~~~~~~

Ensure your PR:

* ✅ Passes all tests
* ✅ Maintains or improves code coverage
* ✅ Follows code style guidelines
* ✅ Includes documentation updates
* ✅ Has a clear, descriptive title
* ✅ Includes a detailed description

PR Description Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Code refactoring

   ## Testing
   Describe the tests you ran and their results.

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests pass
   - [ ] No new warnings

Code Review Process
~~~~~~~~~~~~~~~~~~~

After submission:

1. Automated checks run (tests, linting)
2. Maintainer reviews your code
3. Address any requested changes
4. Once approved, PR is merged

Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

Include:

* Python version
* Package version
* Operating system
* Minimal reproducible example
* Expected vs actual behavior
* Error messages/tracebacks

Feature Requests
~~~~~~~~~~~~~~~~

Include:

* Clear description of the feature
* Use case and motivation
* Proposed implementation (if applicable)
* Examples of similar features elsewhere

Community Guidelines
--------------------

Be Respectful
~~~~~~~~~~~~~

* Use welcoming and inclusive language
* Respect differing viewpoints
* Accept constructive criticism gracefully
* Focus on what's best for the community

Communication
~~~~~~~~~~~~~

* Be clear and concise
* Provide context for your contributions
* Ask questions when unsure
* Help others when you can

Getting Help
------------

If you need help contributing:

* Check existing issues and PRs
* Ask questions in GitHub Discussions
* Contact the maintainer: riccardo.finotello@cea.fr

Recognition
-----------

All contributors are acknowledged in:

* GitHub contributors list
* Release notes
* Project documentation

Thank you for contributing! 🎉
