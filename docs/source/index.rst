HSI Mars: Hyperspectral Image Analysis for Mars Reconnaissance Orbiter
=======================================================================

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.12+

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

A Python package for loading, processing, and visualising hyperspectral imaging (HSI) data from the `CRISM <http://crism.jhuapl.edu/>`_ instrument aboard NASA's Mars Reconnaissance Orbiter (MRO).

Overview
--------

This package provides tools for working with Martian hyperspectral data:

* 📊 Load and process `CRISM <http://crism.jhuapl.edu/>`_ hyperspectral images in ENVI format
* 🏷️ Handle ground truth annotations for machine learning
* 🎨 Create false-colour visualisations of spectral data
* 📈 Plot and analyse spectral signatures
* 📉 Generate histograms for spectral band analysis
* ⚡ Work with large datasets through lazy loading

Dataset
-------

This package is designed to work with the dataset published by Xi *et al.* (2025) in IEEE Geoscience and Remote Sensing Letters.

* **Paper**: `Xi et al., IEEE GRSL, 2025 <https://ieeexplore.ieee.org/document/10843260>`_
* **Dataset**: `SciDB Repository <https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786>`_

Quick Start
-----------

A simple example:

.. code-block:: python

   from hsimars import HSIMars

   # Load a hyperspectral image
   hsi = HSIMars(hdr_path="path/to/your/image.hdr")

   # Get image data and metadata
   img_data = hsi.get_img()
   print(f"Image shape: {img_data.shape}")
   print(f"Wavelength range: {img_data.wavelength.min():.1f} - {img_data.wavelength.max():.1f} nm")

   # Display the false-colour image
   hsi.display_hsi()

   # Plot spectrum for a specific pixel
   hsi.plot_spectra(px=[100, 200], bands=True)

Working with Annotations
~~~~~~~~~~~~~~~~~~~~~~~~

If you have ground truth annotations:

.. code-block:: python

   from hsimars import HSIMars

   # Load image with annotations
   hsi = HSIMars(
       hdr_path="path/to/image.hdr",
       annotations="path/to/labels.mat"
   )

   # Get both image and annotation data
   img_data, ann_data = hsi.data()

   # Display image with annotation overlay
   hsi.display()  # Shows three panels: image, annotations, and overlay

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   modules/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   examples
   contributing

Key Features
------------

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package uses lazy loading for memory efficiency when working with large hyperspectral datasets. Data loads from disk only when first accessed, then caches for later use.

Spectral Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~

* **Spectral Signatures**: Plot individual or averaged spectra with optional convex hull removal for continuum normalisation
* **Band Visualisation**: Overlay spectral band regions (VIS, NIR, SWIR, MWIR) on plots
* **Statistical Analysis**: Generate histograms for analysing spectral band distributions

Visualisation Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **False-Colour Images**: Render HSI data using representative spectral bands
* **Annotation Overlays**: Display ground truth labels with colour-coded classes
* **Interactive Display**: OpenCV-based interactive windows for data exploration

Contributing
------------

Contributions are welcome. To contribute:

1. Fork the repository on GitHub
2. Create a feature branch (``git checkout -b feature/new-feature``)
3. Make your changes
4. Run tests (``pytest tests/``)
5. Commit your changes (``git commit -m 'Add new feature'``)
6. Push to your fork (``git push origin feature/new-feature``)
7. Open a pull request

Ensure your code follows the existing style and includes tests and documentation.

Licence
-------

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
