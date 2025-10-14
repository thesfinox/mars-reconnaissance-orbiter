# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

from hsimars import __author__, __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = (
    "HSIMars - Open hyperspectral images of the Mars Reconnaissance Orbiter"
)
copyright = (
    "2025, Commissariat à l'énergie atomique et aux énergies alternatives"
)
author = __author__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "nbsphinx",  # Jupyter notebook support
]

# Autoclass settings
autoclass_content = "both"

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = False  # Raise errors if notebook has errors
nbsphinx_timeout = 300  # Timeout for notebook execution (if enabled)
nbsphinx_prolog = """
.. note::
   This page was generated from a Jupyter notebook. You can download the notebook
   from the `notebooks/ directory <https://github.com/thesfinox/mars-reconnaissance-orbiter/tree/mars_data/notebooks>`_
   in the repository.
"""

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
