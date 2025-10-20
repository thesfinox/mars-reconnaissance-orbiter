Tutorials
=========

This section provides hands-on tutorials for working with hyperspectral imaging data from Mars. The tutorials are presented as interactive Jupyter notebooks that you can download and run locally.

Getting the Notebooks
---------------------

The tutorial notebooks are in the ``notebooks/`` directory of the repository. You can:

1. **Run them locally** after cloning the repository:

   .. code-block:: bash

      git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git
      cd mars-reconnaissance-orbiter/notebooks
      jupyter notebook  # or jupyter lab

2. **Download individual notebooks** from the `GitHub repository <https://github.com/thesfinox/mars-reconnaissance-orbiter/tree/mars_data/notebooks>`_

3. **Read them below** in the documentation (rendered from the actual notebooks)

Prerequisites
-------------

Before working through the tutorials, ensure you have:

* Installed the HSI Mars package (see :doc:`installation`)
* Basic knowledge of Python programming
* Familiarity with NumPy and matplotlib
* Understanding of basic machine learning concepts (helpful but not required)

Tutorial Notebooks
------------------

The following Jupyter notebooks provide step-by-step tutorials with executable code, visualisations, and detailed explanations.

1. Exploratory Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   notebooks/exploratory_data_analysis

**Target Audience:** Students in introductory machine learning courses

**What you'll learn:**

* Understanding hyperspectral imaging and the `CRISM <http://crism.jhuapl.edu/>`_ instrument on Mars
* Loading and inspecting multidimensional data structures
* Visualising spectral signatures and false-colour images
* Analysing ground truth annotations for supervised learning
* Statistical analysis and data quality assessment
* Preprocessing considerations for machine learning
* Feature engineering in high-dimensional spaces
* The curse of dimensionality and how to address it

**Learning Outcomes:**

After completing this tutorial, you will be able to:

* Load and work with hyperspectral data using the HSI Mars package
* Create publication-quality visualisations of spectral data
* Understand spectral signatures and their role in material identification
* Prepare hyperspectral data for machine learning
* Assess data quality and identify potential issues
* Understand the challenges of high-dimensional remote sensing data

2. Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   notebooks/hsimars

**Target Audience:** All users (beginners to advanced)

**What you'll learn:**

* Quick examples of basic package functionality
* Loading hyperspectral images and accessing metadata
* Creating visualisations and false-colour composites
* Plotting spectral signatures
* Working with ground truth annotations
* Essential HSIMars class methods

**Use this notebook for:**

* Quick reference of common operations
* Testing your installation
* Learning the basic API
* Code snippets you can copy and adapt

Dataset Information
-------------------

The tutorials use data from the Mars Reconnaissance Orbiter's `CRISM <http://crism.jhuapl.edu/>`_ instrument. The dataset was published by:

    Xi *et al.*, "A Large-Scale Benchmark Dataset for Martian Surface Material Classification,"
    IEEE Geoscience and Remote Sensing Letters, 2025.

* **Paper**: https://ieeexplore.ieee.org/document/10843260
* **Dataset**: https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786

The dataset includes:

* Hyperspectral images in ENVI format (.hdr and .img files)
* Ground truth annotations for supervised learning (.mat files)
* Multiple Mars surface sites with different geological features

Troubleshooting
---------------

**ModuleNotFoundError**
   Ensure the package is installed:

   .. code-block:: bash

      uv sync  # if using uv
      # or
      pip install -e .  # if using pip

**OpenCV Window Issues**
   The ``display()`` methods open OpenCV windows. On remote servers:

   * Set up X11 forwarding
   * Comment out ``display()`` calls
   * Use only matplotlib plotting methods

**Memory Issues**
   Hyperspectral data can be large (400+ MB). If you run out of memory:

   * Close other applications
   * Work with a subset of the data
   * Use a machine with more RAM (8GB+ recommended)

**Path Issues**
   Verify data files are in the correct location:

   .. code-block:: text

      mars-reconnaissance-orbiter/
      ├── data/
      │   ├── HC_frt0000580c_07_if164j_ter3.hdr
      │   ├── HC_frt0000580c_07_if164j_ter3.img
      │   └── HC_ground_truth.mat
      └── notebooks/
          ├── exploratory_data_analysis.ipynb
          └── hsimars.ipynb
