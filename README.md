# Hyperspectral Images Exploration

This repository contains tools for the visualisation of hyperspectral images (HSI) of the Mars Reconnaissance Orbiter (MRO) CRISM instrument.
It is based on the [dataset](https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786), which accompanies the [article](https://ieeexplore.ieee.org/document/10843260) by Xi _et al._ (2025).

## Installation

You can install the package as follows (make sure you have Python 3.12+ and `pip` installed):

```bash
pip install git+https://github.com/thesfinox/mars-reconnaissance-orbiter.git
```

or, if you prefer to use `uv`:

```bash
uv install git+https://github.com/thesfinox/mars-reconnaissance-orbiter.git
```

## Development version

You can then clone the repository and install it in editable mode:

```bash
git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git
cd mars-reconnaissance-orbiter
pip install -e .
```

or with `uv`:

```bash
git clone https://github.com/thesfinox/mars-reconnaissance-orbiter.git
cd mars-reconnaissance-orbiter
uv sync
```

## Usage

You can refer to this explanatory [notebook](./notebooks/hsi_hymars.ipynb), or to the documentation of the functions.
