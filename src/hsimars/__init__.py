"""
HSI Mars: Hyperspectral Image Analysis for Mars Reconnaissance Orbiter Data
============================================================================

This package provides tools for loading, processing, and visualizing hyperspectral
imaging (HSI) data from the CRISM instrument aboard NASA's Mars Reconnaissance
Orbiter (MRO). It supports working with ENVI format spectral data and associated
ground truth annotations.

Main Features
-------------
* Load and process CRISM hyperspectral images in ENVI format
* Handle ground truth annotations for supervised learning tasks
* Visualize false-color images and spectral signatures
* Plot spectral profiles with optional convex hull removal
* Generate histograms for spectral band analysis
* Memory-efficient lazy loading for large datasets

Quick Start
-----------
>>> from hsimars import HSIMars
>>> # Load hyperspectral image
>>> hsi = HSIMars(hdr_path="path/to/image.hdr")
>>> img_data = hsi.get_img()
>>> print(f"Image shape: {img_data.shape}")
>>>
>>> # Load with annotations
>>> hsi = HSIMars(
...     hdr_path="path/to/image.hdr", annotations="path/to/labels.mat"
... )
>>> img_data, ann_data = hsi.data()
>>>
>>> # Visualize
>>> hsi.display()  # Interactive display
>>> hsi.plot_spectra(px=[100, 200], convex_hull=True, bands=True)

Classes
-------
HSIMars
    Main class for loading and manipulating hyperspectral images and annotations.

Modules
-------
hsi
    Core module containing the HSIMars class and processing functions.

Package Information
-------------------
Author: Riccardo Finotello
Email: riccardo.finotello@cea.fr
License: GNU General Public License v3 (GPLv3)
URL: https://github.com/thesfinox/mars-reconnaissance-orbiter

See Also
--------
The package is based on the dataset accompanying the article:
Xi et al. (2025), IEEE, DOI: 10.1109/LGRS.2025.3522631
Dataset: https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786
"""

from .hsi import HSIMars

# Package metadata
__version__ = "0.0.1"
__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__license__ = "GNU General Public License v3 (GPLv3)"
__url__ = "https://github.com/thesfinox/mars-reconnaissance-orbiter"

# Public API
__all__ = [
    "HSIMars",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
]
