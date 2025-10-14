"""
Pytest configuration and fixtures for HSIMars testing.

This module provides stub/mock data fixtures for testing the HSIMars class
without requiring real CRISM data files. All test data is generated in memory
or in temporary directories, ensuring tests are fast, isolated, and reproducible.

**Author**: Riccardo Finotello <riccardo.finotello@cea.fr>

**Maintainer**: Riccardo Finotello <riccardo.finotello@cea.fr>

**Contributors**:

    - Riccardo Finotello
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import scipy.io as sio

# Test data dimensions (matching original test dataset expectations)
STUB_HEIGHT = 100
STUB_WIDTH = 120
STUB_CHANNELS = 50
STUB_RAW_HEIGHT = 105  # Raw data before processing (with bad bands)
STUB_RAW_CHANNELS = 55  # Raw channels before bad band removal


class StubPaths(NamedTuple):
    """Container for test file paths."""

    hdr_path: Path
    img_path: Path
    annotations_path: Path
    temp_dir: Path


@pytest.fixture
def stub_wavelengths():
    """Generate realistic wavelength array for CRISM-like data."""
    # CRISM typically covers ~400-4000 nm
    # Generate wavelengths with some non-uniform spacing
    wavelengths = np.linspace(400, 3800, STUB_RAW_CHANNELS)
    return wavelengths.astype(np.float32)


@pytest.fixture
def stub_bad_bands():
    """Generate bad band list (indices to remove during processing)."""
    # Simulate 5 bad bands that will be removed
    # This explains why STUB_RAW_CHANNELS (55) becomes STUB_CHANNELS (50)
    return [5, 12, 25, 38, 49]


@pytest.fixture
def stub_envi_header(tmp_path, stub_wavelengths, stub_bad_bands):
    """
    Create a stub ENVI header (.hdr) file with realistic metadata.

    The header file contains all necessary information for reading the
    binary image data, including dimensions, wavelengths, and bad bands.
    """
    hdr_path = tmp_path / "test_data.hdr"

    # Create ENVI header content
    # Select default bands for visualization (avoiding bad bands)
    #  Choose bands spread across the spectrum
    default_band_indices = [10, 25, 40]  # RGB-like bands for visualization

    header_content = f"""ENVI
description = {{Stub CRISM data for testing}}
samples = {STUB_WIDTH}
lines = {STUB_RAW_HEIGHT}
bands = {STUB_RAW_CHANNELS}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bil
byte order = 0
data ignore value = 65535
default bands = {{{", ".join(str(b) for b in default_band_indices)}}}
wavelength = {{
 {", ".join(f"{w:.6f}" for w in stub_wavelengths)}
}}
wavelength units = nm
bbl = {{
 {", ".join("1" if i not in stub_bad_bands else "0" for i in range(STUB_RAW_CHANNELS))}
}}
"""

    hdr_path.write_text(header_content)
    return hdr_path


@pytest.fixture
def stub_envi_image(tmp_path, stub_envi_header):
    """
    Create a stub ENVI binary image (.img) file with synthetic HSI data.

    Generates random but realistic hyperspectral data with spatial and
    spectral structure. The data is saved in BIL (Band Interleaved by Line)
    format as specified in the header.
    """
    img_path = tmp_path / "test_data.img"

    # Generate synthetic hyperspectral data
    # Shape: (lines, bands, samples) for BIL format
    # Create data with some spatial structure
    np.random.seed(42)  # Reproducible random data

    # Create base patterns
    x = np.linspace(0, 4 * np.pi, STUB_WIDTH)
    y = np.linspace(0, 4 * np.pi, STUB_RAW_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    # Generate data with spatial patterns
    data = np.zeros(
        (STUB_RAW_HEIGHT, STUB_RAW_CHANNELS, STUB_WIDTH), dtype=np.float32
    )

    for band in range(STUB_RAW_CHANNELS):
        # Create patterns that vary by band
        pattern = (
            np.sin(xx + band * 0.1) * np.cos(yy + band * 0.1)
            + np.random.randn(STUB_RAW_HEIGHT, STUB_WIDTH) * 0.1
        )
        # Normalize to realistic reflectance values (0-1 range)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        data[:, band, :] = pattern * 0.8 + 0.1  # Scale to 0.1-0.9 range

    # Write binary data (BIL format)
    data.tofile(str(img_path))

    return img_path


@pytest.fixture
def stub_annotations(tmp_path, stub_envi_header):
    """
    Create stub ground truth annotations in MATLAB .mat format.

    Generates synthetic label data with multiple classes arranged in
    spatial regions. Labels are stored in the 'map' field to match
    the expected structure of CRISM annotation files.
    """
    annotations_path = tmp_path / "ground_truth.mat"

    # Create synthetic label data with spatial structure
    # Use processed dimensions (after bad band removal and cropping)
    labels = np.zeros((STUB_HEIGHT, STUB_WIDTH), dtype=np.uint8)

    # Create regions for different classes (0 = background, 1-4 = classes)
    # Class 1: Top-left quadrant
    labels[: STUB_HEIGHT // 2, : STUB_WIDTH // 2] = 1

    # Class 2: Top-right quadrant
    labels[: STUB_HEIGHT // 2, STUB_WIDTH // 2 :] = 2

    # Class 3: Bottom-left quadrant with some variation
    labels[STUB_HEIGHT // 2 :, : STUB_WIDTH // 2] = 3

    # Class 4: Bottom-right quadrant (partial)
    labels[
        STUB_HEIGHT // 2 + 10 : STUB_HEIGHT - 10,
        STUB_WIDTH // 2 + 10 : STUB_WIDTH - 10,
    ] = 4

    # Save as MATLAB file with 'map' field (expected structure)
    sio.savemat(str(annotations_path), {"map": labels})

    return annotations_path


@pytest.fixture
def stub_hsi_files(
    tmp_path, stub_envi_header, stub_envi_image, stub_annotations
):
    """
    Convenience fixture that provides all stub data file paths.

    Returns a NamedTuple with paths to all generated test files and
    the temporary directory.
    """
    return StubPaths(
        hdr_path=stub_envi_header,
        img_path=stub_envi_image,
        annotations_path=stub_annotations,
        temp_dir=tmp_path,
    )


@pytest.fixture
def stub_hsi_no_annotations(tmp_path, stub_envi_header, stub_envi_image):
    """
    Fixture for HSI data without annotations.

    Useful for testing scenarios where only image data is available.
    """
    return StubPaths(
        hdr_path=stub_envi_header,
        img_path=stub_envi_image,
        annotations_path=None,
        temp_dir=tmp_path,
    )


@pytest.fixture
def expected_stub_dimensions():
    """Fixture providing expected dimensions for stub data after processing.

    Note: Since stub data contains no invalid pixels (value 65535),
    the image is not cropped during processing, so height remains at raw height.
    """
    return {
        "height": STUB_RAW_HEIGHT,  # No cropping occurs with valid stub data
        "width": STUB_WIDTH,
        "channels": STUB_CHANNELS,
        "raw_height": STUB_RAW_HEIGHT,
        "raw_channels": STUB_RAW_CHANNELS,
    }


@pytest.fixture
def nonexistent_file_path(tmp_path):
    """Fixture providing path to a non-existent file for error testing."""
    return tmp_path / "nonexistent_file.hdr"


@pytest.fixture
def stub_label_names_excel(tmp_path):
    """
    Create a stub Excel file with label name mappings.

    Creates an Excel file matching the expected format with test dataset
    information. The file contains label mappings for a 'TE' (test) dataset.
    """
    import pandas as pd

    excel_path = tmp_path / "data_description.xlsx"

    # Create data matching the expected format
    # The stub data uses 'TE' as the prefix (from 'test_data.hdr')
    data = [
        ["TE", None, None],
        ["Class  ID", "Class Name", "Total"],
        [1, "Test Mineral A", 3000],
        [2, "Test Mineral B", 3000],
        [3, "Test Mineral C", 3000],
        [4, "Test Mineral D", 1200],
        [None, "Total", 10200],
    ]

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False, header=False)

    return excel_path
