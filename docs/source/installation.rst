Installation Guide
==================

This guide will walk you through the installation process for the HSI Mars package.

Prerequisites
-------------

Before installing this package, ensure you have:

* **Python 3.12 or higher** installed on your system

  * Download from `python.org <https://www.python.org/downloads/>`_
  * Verify installation: ``python --version``

Installation Methods
--------------------

We provide two main installation methods: using ``uv`` (recommended) or using ``pip`` (traditional).

Using uv (recommended)
~~~~~~~~~~~~~~~~~~~~~~

`uv <https://docs.astral.sh/uv/>`_ is a fast, modern Python package manager that simplifies dependency management. We recommend using ``uv`` for the best experience.

Step 1: Install uv
^^^^^^^^^^^^^^^^^^

If you haven't installed ``uv`` yet:

**On macOS and Linux:**

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

**On Windows (PowerShell):**

.. code-block:: powershell

   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

For alternative installation methods, see the `uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

Step 2: Install the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Option A: Install from GitHub** (for end users)

.. code-block:: bash

   mkdir hsi_mars && cd hsi_mars
   uv init --python=3.12
   uv add git+https://github.com/thesfinox/mars-reconnaissance-orbiter.git

**Option B: Install for Development** (for contributors)

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git hsi_mars
   cd hsi_mars

   # Create a virtual environment and install dependencies
   uv sync

   # Install with development dependencies (optional)
   uv sync --group dev

   # Activate the virtual environment
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate  # On Windows

Using pip
~~~~~~~~~

If you prefer the traditional pip workflow:

Step 1: Create a Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create a new virtual environment
   mkdir hsi_mars && cd hsi_mars
   python -m venv .venv

   # Activate the virtual environment
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate  # On Windows

   # Upgrade pip to the latest version
   pip install --upgrade pip

Step 2: Install the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Option A: Install from GitHub** (for end users)

.. code-block:: bash

   pip install git+https://github.com/thesfinox/mars-reconnaissance-orbiter.git

**Option B: Install for Development** (for contributors)

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git
   cd mars-reconnaissance-orbiter

   # Install in editable mode
   pip install -e .

   # Or install with development dependencies
   pip install -e ".[dev]"

Development Setup
-----------------

For contributors who want to set up a complete development environment:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git
   cd mars-reconnaissance-orbiter

   # Using uv (recommended)
   uv sync --group dev --group test --group docs

   # Or using pip
   pip install -e ".[dev,test,docs]"

   # Install pre-commit hooks
   pre-commit install

   # Run tests
   pytest tests/

   # Build documentation
   cd docs
   make html

Verifying Installation
----------------------

To verify that the package is installed correctly, run:

.. code-block:: python

   python -c "from hsimars import HSIMars; print(HSIMars.__module__)"

If this command runs without errors, the installation was successful!

Troubleshooting
---------------

Python Version Issues
~~~~~~~~~~~~~~~~~~~~~

If you encounter issues related to Python version:

1. Ensure you have Python 3.12 or higher installed
2. You may need to use ``python3.12`` instead of ``python`` in commands
3. Consider using `pyenv <https://github.com/pyenv/pyenv>`_ to manage multiple Python versions

Installation Failures
~~~~~~~~~~~~~~~~~~~~~

If installation fails:

1. Ensure your pip is up to date: ``pip install --upgrade pip``
2. Try installing in a fresh virtual environment
3. Check that you have sufficient permissions (avoid using ``sudo`` with pip)
4. On Linux, you may need to install development headers: ``sudo apt-get install python3-dev``

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~

If you encounter dependency conflicts:

1. Try installing in a fresh virtual environment
2. Use ``uv`` which handles dependency resolution more robustly
3. Check the ``pyproject.toml`` file for specific dependency versions

Getting Help
~~~~~~~~~~~~

If you continue to experience issues:

* Check the `GitHub Issues <https://github.com/thesfinox/mars-reconnaissance-orbiter/issues>`_
* Open a new issue with details about your system and the error message
* Contact the maintainer: riccardo.finotello@cea.fr

Next Steps
----------

Now that you have installed the package, proceed to the :doc:`quickstart` guide to learn how to use it!
