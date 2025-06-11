"""
Test the HSImage class.

Author: Riccardo Finotello <riccardo.finotello@cea.fr>
"""

import pytest

from hsimars import HSIMars


class TestHSImage:
    """Test the HSImage class."""

    def test_constructor(self):
        """Test the constructor class."""

        with pytest.raises(FileNotFoundError, match="header file"):
            HSIMars(hdr_path="does-not-exist.hdr")

        with pytest.raises(FileNotFoundError, match="annotation file"):
            HSIMars(
                hdr_path="data/UP_frt0000527d_07_if166j_ter3.hdr",
                annotations="does-not-exist.mat",
            )

        hsi = HSIMars(
            hdr_path="data/UP_frt0000527d_07_if166j_ter3.hdr",
            annotations="data/UP_ground_truth.mat",
        )
        assert str(hsi.hdr_path) == "data/UP_frt0000527d_07_if166j_ter3.hdr"
        assert str(hsi.annotations) == "data/UP_ground_truth.mat"

    def test_raw(self):
        """Test the raw data."""

        hsi = HSIMars(hdr_path="data/UP_frt0000527d_07_if166j_ter3.hdr")
        assert hsi._raw is None
        raw = hsi.get_raw()
        assert hsi._raw == raw

    def test_img(self):
        """Test the image manipulation."""

        hsi = HSIMars(hdr_path="data/UP_frt0000527d_07_if166j_ter3.hdr")
        assert hsi._img is None
        img = hsi.get_img()
        assert hsi._img == img
        assert hasattr(img, "hsi")
        assert hasattr(img, "wavelength")
        assert hasattr(img, "shape")
        assert img.shape == (478, 600, 465)
        assert hasattr(img, "height")
        assert img.height == 478
        assert hasattr(img, "width")
        assert img.width == 600
        assert hasattr(img, "channels")
        assert img.channels == 465
        assert len(img.wavelength) == 465
        assert hasattr(img, "dtype")
        assert img.dtype == "float32"

    def test_ann(self):
        """Test the annotation retrieval."""

        hsi = HSIMars(
            hdr_path="data/UP_frt0000527d_07_if166j_ter3.hdr",
            annotations="data/UP_ground_truth.mat",
        )
        assert hsi._ann is None
        ann = hsi.get_annotations()
        assert hsi._ann == ann
        assert hasattr(ann, "labels")
        assert hasattr(ann, "shape")
        assert ann.shape == (478, 600)
        assert hasattr(ann, "height")
        assert ann.height == 478
        assert hasattr(ann, "width")
        assert ann.width == 600
        assert hasattr(ann, "dtype")
        assert ann.dtype == "uint8"

    def test_data(self):
        """Test the all inclusive methods."""

        hsi = HSIMars(
            hdr_path="data/UP_frt0000527d_07_if166j_ter3.hdr",
            annotations="data/UP_ground_truth.mat",
        )
        assert hsi._raw is None
        assert hsi._img is None
        assert hsi._ann is None
        img, ann = hsi.data()
        assert hsi._img == img
        assert hasattr(img, "hsi")
        assert hasattr(img, "wavelength")
        assert hasattr(img, "shape")
        assert img.shape == (478, 600, 465)
        assert hasattr(img, "height")
        assert img.height == 478
        assert hasattr(img, "width")
        assert img.width == 600
        assert hasattr(img, "channels")
        assert img.channels == 465
        assert len(img.wavelength) == 465
        assert hasattr(img, "dtype")
        assert img.dtype == "float32"
        assert hsi._ann == ann
        assert hasattr(ann, "labels")
        assert hasattr(ann, "shape")
        assert ann.shape == (478, 600)
        assert hasattr(ann, "height")
        assert ann.height == 478
        assert hasattr(ann, "width")
        assert ann.width == 600
        assert hasattr(ann, "dtype")
        assert ann.dtype == "uint8"
