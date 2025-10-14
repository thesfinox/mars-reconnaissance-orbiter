# HSI Mars: Hyperspectral Image Analysis for Mars Reconnaissance Orbiter

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package for loading, processing, and visualizing hyperspectral imaging (HSI) data from the CRISM instrument aboard NASA’s Mars Reconnaissance Orbiter (MRO).

## Overview

This package provides intuitive tools for working with Martian hyperspectral data, enabling researchers to:

- 📊 Load and process CRISM hyperspectral images in ENVI format
- 🏷️ Handle ground truth annotations for machine learning applications
- 🎨 Create false-color visualizations of spectral data
- 📈 Plot and analyze spectral signatures with advanced processing
- 📉 Generate histograms for spectral band analysis
- ⚡ Work efficiently with large datasets through lazy loading

## Dataset

This package is designed to work with the dataset published by Xi _et al._ (2025) in IEEE Geoscience and Remote Sensing Letters.

- **Paper**: [Xi et al., IEEE GRSL, 2025](https://ieeexplore.ieee.org/document/10843260)
- **Dataset**: [SciDB Repository](https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before installing this package, ensure you have:

- **Python 3.12 or higher** installed on your system
  - Download from [python.org](https://www.python.org/downloads/)
  - Verify installation: `python --version`

## Installation

### Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast, modern Python package manager that simplifies dependency management. We recommend using `uv` for the best experience.

#### Step 1: Install uv

If you haven’t installed `uv` yet:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For alternative installation methods, see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

#### Step 2: Install the package

**Option A: Install from GitHub** (for end users)

```bash
mkdir hsi_mars && cd hsi_mars
uv init --python=3.12
uv add git+https://github.com/thesfinox/mars-reconnaissance-orbiter.git
```

**Option B: Install for development** (for contributors)

```bash
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
```

### Using pip

If you prefer the traditional pip workflow:

#### Step 1: Create a virtual environment

```bash
# Create a new virtual environment
mkdir hsi_mars && cd hsi_mars
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Upgrade pip to the latest version
pip install --upgrade pip
```

#### Step 2: Install the package

**Option A: Install from GitHub** (for end users)

```bash
pip install git+https://github.com/thesfinox/mars-reconnaissance-orbiter.git
```

**Option B: Install for development** (for contributors)

```bash
# Clone the repository
git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git
cd mars-reconnaissance-orbiter

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

Here’s a simple example to get you started:

```python
from hsimars import HSIMars

# Load a hyperspectral image
hsi = HSIMars(hdr_path="path/to/your/image.hdr")

# Get image data and metadata
img_data = hsi.get_img()
print(f"Image shape: {img_data.shape}")
print(f"Wavelength range: {img_data.wavelength.min():.1f} - {img_data.wavelength.max():.1f} nm")

# Display the false-color image
hsi.display_hsi()

# Plot spectrum for a specific pixel
hsi.plot_spectra(px=[100, 200], bands=True)
```

### Working with Annotations

If you have ground truth annotations:

```python
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
```

## Development Setup

For contributors who want to set up a complete development environment:

```bash
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
```

## Usage Examples

### Analyzing Multiple Pixels

```python
# Plot average spectrum from a region
pixels = [[100, 200], [101, 200], [100, 201], [101, 201]]
hsi.plot_spectra(px=pixels, convex_hull=True, bands=True)
```

### Generating Histograms

```python
# Histogram for a specific wavelength
hsi.plot_histogram(band=1500.0)  # 1500 nm

# Or by band index
hsi.plot_histogram(band=100)
```

### Saving Plots

```python
# Save spectrum plot
hsi.plot_spectra(
    px=[100, 200],
    bands=True,
    output="results/spectrum.png"
)

# Save histogram
hsi.plot_histogram(
    band=1500.0,
    output="results/histogram_1500nm.png"
)
```

## Documentation

For detailed API documentation and advanced usage:

- **API Reference**: Check the docstrings in the source code
- **Jupyter Notebook**: See [`notebooks/hsi_hymars.ipynb`](./notebooks/hsi_hymars.ipynb) for interactive examples
- **Sphinx Documentation**: Run `make html` in the `docs/` directory

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests and documentation.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
