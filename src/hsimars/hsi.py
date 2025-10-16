"""
Open and manipulate hyperspectral images (HSI) from Mars Reconnaissance Orbiter.

This module provides the `HSIMars` class for working with hyperspectral imaging data
from the CRISM instrument aboard the Mars Reconnaissance Orbiter. It supports loading,
processing, and visualizing HSI data along with their annotations.

The module handles:
- Loading ENVI format hyperspectral images (.hdr and .img files)
- Processing and cleaning HSI data (removing bad bands, cropping, normalization)
- Loading and aligning ground truth annotations (.mat files)
- Visualizing spectral data, false-color images, and annotations
- Plotting spectra with optional convex hull removal
- Generating histograms for specific spectral bands

**Author**: Riccardo Finotello <riccardo.finotello@cea.fr>

**Maintainer**: Riccardo Finotello <riccardo.finotello@cea.fr>

**Contributors**:

    - Riccardo Finotello

Examples
--------
>>> from hsimars import HSIMars
>>> hsi = HSIMars(
...     hdr_path="path/to/image.hdr", annotations="path/to/annotations.mat"
... )
>>> img_data = hsi.get_img()
>>> print(f"Image shape: {img_data.shape}")
>>> hsi.plot_spectra(px=[100, 200], convex_hull=True, bands=True)
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pysptools.spectro import convex_hull_removal
from scipy.interpolate import splev, splrep
from scipy.io import loadmat
from spectral.io import envi

# Configure matplotlib for consistent styling
plt.style.use("grayscale")
mpl.rc("font", size=16)


class HSIMars:
    """
    Load and manipulate hyperspectral images from Mars Reconnaissance Orbiter.

    This class provides comprehensive functionality for working with CRISM
    hyperspectral imaging data, including loading ENVI format images, processing
    spectral data, handling ground truth annotations, and creating visualizations.

    The class implements lazy loading to optimize memory usage - data is only loaded
    from disk when first accessed and then cached for subsequent operations.

    Attributes
    ----------
    hdr_path : Path
        Path to the ENVI header file (.hdr extension).
    annotations : Path or None
        Path to the ground truth annotations file (.mat format), if provided.

    Examples
    --------
    >>> # Load HSI data without annotations
    >>> hsi = HSIMars(hdr_path="data/sample.hdr")
    >>> img_data = hsi.get_img()
    >>> print(
    ...     f"Image dimensions: {img_data.height}x{img_data.width}, {img_data.channels} channels"
    ... )

    >>> # Load HSI data with annotations
    >>> hsi = HSIMars(hdr_path="data/sample.hdr", annotations="data/labels.mat")
    >>> img_data, ann_data = hsi.data()
    >>> hsi.display()  # Show image with overlaid annotations

    >>> # Plot spectrum for a specific pixel
    >>> hsi.plot_spectra(px=[100, 200], convex_hull=True, bands=True)

    Notes
    -----
    The ENVI .img file containing the actual hyperspectral data must be located
    in the same directory as the .hdr header file.
    """

    def __init__(
        self,
        hdr_path: str | Path,
        annotations: str | Path | None = None,
        label_names_path: str | Path | None = None,
    ):
        """
        Initialize the HSIMars object with paths to data files.

        Parameters
        ----------
        hdr_path : str | Path
            Path to the ENVI header file (.hdr extension) containing metadata
            about the hyperspectral image. The corresponding .img file with the
            actual spectral data must be in the same directory.
        annotations : str | Path, optional
            Path to the ground truth annotations file (.mat format). If None,
            annotation-related methods will return None. Default is None.
        label_names_path : str | Path, optional
            Path to the Excel file containing label name mappings. If None,
            the method will look for a file named ``data_description.xlsx``
            in the same directory as the HDR file. Default is None.

        Raises
        ------
        FileNotFoundError
            If the header file does not exist at the specified path.
        FileNotFoundError
            If annotations path is provided but the file does not exist.
        FileNotFoundError
            If label_names_path is provided but the file does not exist.

        Examples
        --------
        >>> hsi = HSIMars(hdr_path="path/to/image.hdr")
        >>> hsi_with_labels = HSIMars(
        ...     hdr_path="path/to/image.hdr", annotations="path/to/labels.mat"
        ... )
        >>> hsi_custom_labels = HSIMars(
        ...     hdr_path="path/to/image.hdr",
        ...     annotations="path/to/labels.mat",
        ...     label_names_path="path/to/custom_labels.xlsx",
        ... )

        Notes
        -----
        The constructor only validates file paths - no data is loaded until
        one of the get_* methods is called. This lazy loading approach
        minimizes memory usage when working with large datasets.
        """
        self.hdr_path: Path = Path(hdr_path)
        if not self.hdr_path.exists():
            raise FileNotFoundError(
                f"The header file {self.hdr_path} could not be found in the filesystem."
            )
        self.annotations: Path | None = None
        if annotations is not None:
            self.annotations = Path(annotations)
            if not self.annotations.exists():
                raise FileNotFoundError(
                    f"The annotation file {self.annotations} could not be found in the filesystem."
                )

        self.label_names_path: Path | None = None
        if label_names_path is not None:
            self.label_names_path = Path(label_names_path)
            if not self.label_names_path.exists():
                raise FileNotFoundError(
                    f"The label names file {self.label_names_path} could not be found in the filesystem."
                )

        # Cache large objects to avoid repeated disk I/O and processing
        # These are populated on first access via get_* methods
        self._raw: envi.BsqFile | None = None
        self._img: NamedTuple | None = None
        self._ann: NamedTuple | None = None
        self._false_colour_bands: NDArray | None = None

    def get_raw(self) -> envi.BsqFile:
        """
        Load the raw ENVI hyperspectral data file.

        This method opens the ENVI format file and returns a file object that
        provides access to the raw spectral data. The data is cached after the
        first call to avoid redundant disk I/O operations.

        Returns
        -------
        envi.BsqFile
            ENVI file object providing access to the hyperspectral data. This
            object supports memory-mapped access to efficiently handle large files.

        Examples
        --------
        >>> hsi = HSIMars(hdr_path="data/sample.hdr")
        >>> raw = hsi.get_raw()
        >>> print(raw.metadata["wavelength"])  # Access wavelength information

        Notes
        -----
        The actual spectral data is stored in a .img file that must be located
        in the same directory as the .hdr file. The ENVI library automatically
        locates and opens the corresponding .img file.
        """
        if self._raw is None:
            self._raw = envi.open(self.hdr_path)
        return self._raw

    def get_img(self) -> NamedTuple:
        """
        Load and process the hyperspectral image data.

        This method loads the raw HSI data, performs preprocessing steps including
        bad band removal, cropping of invalid pixels, and normalization. The result
        is cached for efficient subsequent access.

        The preprocessing pipeline includes:
        1. Loading wavelength information and identifying bad bands
        2. Removing pixels with the ignore value (65535)
        3. Cropping the image to remove rows/columns with no valid data
        4. Removing any remaining bad channels
        5. Converting to float32 format for numerical processing

        Returns
        -------
        NamedTuple
            A named tuple (HSIMarsImageData) containing the following attributes:

            - **hsi** : ndarray of shape (height, width, channels)
                The processed hyperspectral image data as a 3D numpy array.
                Data type is float32.
            - **wavelength** : ndarray of shape (channels,)
                Array of wavelength values in nanometers corresponding to each
                spectral channel.
            - **shape** : tuple of (height, width, channels)
                Dimensions of the hyperspectral image.
            - **height** : int
                Number of pixel rows in the image.
            - **width** : int
                Number of pixel columns in the image.
            - **channels** : int
                Number of spectral bands/channels in the image.
            - **dtype** : str
                Data type of the HSI array ('float32').

        Examples
        --------
        >>> hsi = HSIMars(hdr_path="data/sample.hdr")
        >>> img_data = hsi.get_img()
        >>> print(f"Image shape: {img_data.shape}")
        >>> print(
        ...     f"Wavelength range: {img_data.wavelength.min():.1f} - {img_data.wavelength.max():.1f} nm"
        ... )
        >>> print(f"Data type: {img_data.dtype}")
        >>> # Access specific pixel spectrum
        >>> spectrum = img_data.hsi[100, 200, :]
        >>> print(f"Spectrum at (100, 200): {spectrum.shape[0]} bands")

        Notes
        -----
        The method implements lazy evaluation - the image is only loaded and
        processed on the first call. Subsequent calls return the cached result.

        The bad band list (bbl) from the ENVI metadata is used to filter out
        unreliable spectral channels before further processing.
        """
        if self._img is not None:
            return self._img

        # Get the raw data
        img = self.get_raw()

        # Recover the image and the metadata
        img_memmap = img.open_memmap()
        img_metadata = img.metadata

        # Extract wavelength information and ignore value for invalid pixels
        wl = np.array(img_metadata["wavelength"]).astype("float32")
        ignore_value = float(img_metadata["data ignore value"])

        # Select the default bands for false-color visualization
        bands = np.array(img_metadata["default bands"]).astype(int)
        bands = wl[bands]

        # Filter out bad bands based on metadata bad band list (bbl)
        bbl = np.array(img_metadata["bbl"]).astype("bool")
        img_memmap = img_memmap[..., bbl]
        wl = wl[bbl]

        # Crop the image by removing rows/columns containing only ignore values
        # This reduces memory usage and eliminates invalid border regions
        mask = img_memmap != ignore_value
        mask_channels = mask.sum(axis=2)
        good_rows = mask_channels.sum(axis=1) > 0
        good_cols = mask_channels.sum(axis=0) > 0
        img_memmap = img_memmap[good_rows, :][:, good_cols]

        # Find and remove any remaining bad channels that still contain
        # ignore values
        mask = img_memmap == ignore_value
        idx = np.unique(np.argwhere(mask)[..., 2])
        ch = np.ones((img_memmap.shape[2],), dtype="bool")
        ch[idx] = False
        img_memmap = img_memmap[..., ch]
        wl = wl[ch]

        # Update the default bands indices for false-color visualization
        self._false_colour_bands = np.searchsorted(wl, bands).astype(int)

        output = namedtuple(
            "HSIMarsImageData",
            [
                "hsi",
                "wavelength",
                "shape",
                "height",
                "width",
                "channels",
                "dtype",
            ],
        )
        self._img = output(
            hsi=img_memmap.astype("float32"),
            wavelength=wl,
            shape=img_memmap.shape,
            height=img_memmap.shape[0],
            width=img_memmap.shape[1],
            channels=img_memmap.shape[2],
            dtype="float32",
        )

        return self._img

    def _pad_annotations(self, mat: NDArray) -> NDArray:
        """
        Pad annotation matrix to match the processed HSI dimensions.

        The annotation matrix may have different dimensions than the processed
        HSI data after cropping. This method centers the annotations within
        the target dimensions by adding symmetric padding.

        Parameters
        ----------
        mat : NDArray
            The raw annotation matrix to be padded.

        Returns
        -------
        NDArray
            The padded annotation matrix matching HSI dimensions.

        Notes
        -----
        Padding is distributed symmetrically around the annotation data,
        with any odd remainder added to the bottom/right edges.
        """
        # Get target shape from processed image
        img = self.get_img()
        H, W = img.height, img.width
        h, w = mat.shape

        # Calculate symmetric padding for height and width
        pad_top = (H - h) // 2
        pad_bottom = H - h - pad_top
        pad_left = (W - w) // 2
        pad_right = W - w - pad_left

        return np.pad(mat, ((pad_top, pad_bottom), (pad_left, pad_right)))

    def _load_label_names(self) -> dict[int, str]:
        """
        Load label names from the data description Excel file.

        This method reads the data description Excel file and extracts the mapping
        between numerical class labels and their human-interpretable names for the
        current dataset.

        The method automatically identifies the dataset by extracting a two-letter
        prefix from the HDR filename (e.g., "HC", "NF", or "UP") and locates the
        corresponding section in the Excel file. It then parses the class ID and
        class name columns to build the label mapping dictionary.

        Returns
        -------
        dict[int, str]
            A dictionary mapping numerical class labels (integers) to their
            corresponding human-readable class names (strings). For example:
            {1: 'Analcime', 2: 'Plagioclase', 3: 'Prehnite', ...}

            Returns an empty dictionary if:
            - The Excel file is not found
            - The dataset prefix is not found in the Excel file
            - The Excel file structure is invalid or cannot be parsed

        Notes
        -----
        If ``label_names_path`` was provided during initialization, that file
        will be used. Otherwise, the method looks for a file named
        ``data_description.xlsx`` in the same directory as the HDR file.

        The file should contain sections for each dataset, where each section
        starts with a row containing the dataset name (e.g., "HC", "NF", "UP"),
        followed by a header row with columns "Class  ID", "Class Name", and
        "Total", and then data rows containing the class ID and name pairs.

        This method is called internally by :meth:`get_annotations` to populate
        the ``label_names`` field of the ``HSIMarsAnnotationData`` named tuple.

        Examples
        --------
        The expected Excel file structure for a dataset section:

        .. code-block:: text

            HC
            Class  ID  |  Class Name       |  Total
            1          |  Analcime         |  940
            2          |  Plagioclase      |  1472
            3          |  Prehnite         |  1560
            ...

        Warnings
        --------
        If the Excel file or dataset section cannot be found, this method
        returns an empty dictionary rather than raising an exception. This
        allows the annotation loading process to continue even when label
        names are unavailable, though the ``label_names`` field will be empty.
        """
        # Determine which Excel file to use
        if self.label_names_path is not None:
            excel_path = self.label_names_path
        else:
            # Look for data_description.xlsx in same directory as HDR file
            excel_path = self.hdr_path.parent / "data_description.xlsx"

        # Return empty dict if Excel file doesn't exist
        if not excel_path.exists():
            return {}

        # Extract dataset prefix from HDR filename (first 2 characters)
        dataset_name = self.hdr_path.stem[:2].upper()

        try:
            # Read Excel file without header to access raw cell data
            df = pd.read_excel(excel_path, header=None)

            # Find the row containing the dataset name identifier
            dataset_row_idx = None
            for idx, row in df.iterrows():
                if row[0] == dataset_name:
                    dataset_row_idx = idx
                    break

            # Return empty dict if dataset not found in Excel
            if dataset_row_idx is None:
                return {}

            # Skip the header row (next row after dataset name)
            # and start reading class labels from the following row
            data_start_row = dataset_row_idx + 2

            # Parse class ID and name pairs until hitting end marker
            label_map = {}
            for row_idx in range(data_start_row, len(df)):
                row = df.iloc[row_idx]

                # Stop at empty row or "Total" row (end of section)
                if pd.isna(row[0]):
                    break

                # Parse and store class ID to name mapping
                try:
                    class_id = int(row[0])
                    class_name = str(row[1]).strip()
                    label_map[class_id] = class_name
                except (ValueError, TypeError):
                    # Skip rows that don't contain valid ID/name pairs
                    continue

            return label_map

        except Exception:
            # Return empty dict if any error occurs during parsing
            return {}

    def get_annotations(self) -> NamedTuple | None:
        """
        Load and process ground truth annotation data.

        Loads annotation labels from a MATLAB .mat file and aligns them with
        the processed HSI data dimensions. The result is cached for efficient
        subsequent access.

        Returns
        -------
        NamedTuple or None
            If annotations were provided during initialization, returns a named
            tuple (HSIMarsAnnotationData) with the following attributes:

            - **labels** : ndarray of shape (height, width)
                2D array containing class labels for each pixel. Values are
                unsigned integers representing different material classes.
                A value of 0 typically indicates unlabeled/background pixels.
            - **shape** : tuple of (height, width)
                Dimensions of the annotation matrix.
            - **height** : int
                Number of pixel rows (matches HSI height).
            - **width** : int
                Number of pixel columns (matches HSI width).
            - **dtype** : str
                Data type of the labels array ('uint8').
            - **label_names** : dict[int, str]
                Dictionary mapping numerical class labels to human-readable
                class names. For example: {1: 'Analcime', 2: 'Plagioclase'}.
                Will be an empty dictionary if the label mapping file is not
                found or cannot be parsed.

            Returns None if no annotation file was provided during initialization.

        Examples
        --------
        >>> hsi = HSIMars(
        ...     hdr_path="data/sample.hdr", annotations="data/labels.mat"
        ... )
        >>> ann_data = hsi.get_annotations()
        >>> if ann_data is not None:
        ...     print(f"Annotation shape: {ann_data.shape}")
        ...     unique_labels = np.unique(ann_data.labels)
        ...     print(f"Number of classes: {len(unique_labels)}")

        Notes
        -----
        The annotation matrix is automatically padded to match the dimensions
        of the processed HSI data. This ensures pixel-level alignment between
        spectral data and labels.

        The method implements lazy evaluation - annotations are only loaded
        on the first call. Subsequent calls return the cached result.
        """
        if self._ann is not None:
            return self._ann

        if self.annotations is None:
            return None

        # Load the annotation matrix from MATLAB file
        ann = loadmat(self.annotations)
        # Extract the first numpy array found in the .mat file
        mat = None
        for v in ann.values():
            if isinstance(v, np.ndarray):
                mat = v.astype("uint8")
                break

        if mat is None:
            raise ValueError(
                f"No valid annotation matrix found in {self.annotations}"
            )

        # Pad the annotation matrix to match the processed image dimensions
        mat = self._pad_annotations(mat)

        # Load label names from the data description Excel file
        label_names = self._load_label_names()

        output = namedtuple(
            "HSIMarsAnnotationData",
            ["labels", "shape", "height", "width", "dtype", "label_names"],
        )
        self._ann = output(
            labels=mat,
            shape=mat.shape,
            height=mat.shape[0],
            width=mat.shape[1],
            dtype="uint8",
            label_names=label_names,
        )
        return self._ann

    def data(self) -> tuple[NamedTuple, NamedTuple | None]:
        """
        Load both hyperspectral image and annotation data.

        This convenience method loads both the HSI data and annotations (if available)
        in a single call, ensuring both are cached for subsequent operations.

        Returns
        -------
        tuple[NamedTuple, NamedTuple or None]
            A tuple containing two elements:

            1. **HSIMarsImageData** (NamedTuple):

               - **hsi** : ndarray
                   The HSI data array of shape (height, width, channels).
               - **wavelength** : ndarray
                   Array of wavelength values in nm.
               - **shape** : tuple
                   Dimensions (height, width, channels).
               - **height** : int
                   Number of pixel rows.
               - **width** : int
                   Number of pixel columns.
               - **channels** : int
                   Number of spectral bands.
               - **dtype** : str
                   Data type ('float32').

            2. **HSIMarsAnnotationData** (NamedTuple or None):

               If annotations are available:

               - **labels** : ndarray
                   Label array of shape (height, width).
               - **shape** : tuple
                   Dimensions (height, width).
               - **height** : int
                   Number of pixel rows.
               - **width** : int
                   Number of pixel columns.
               - **dtype** : str
                   Data type ('uint8').
               - **label_names** : dict[int, str]
                   Dictionary mapping numerical class labels to human-readable
                   class names.

               Returns None if no annotations were provided.

        Examples
        --------
        >>> hsi = HSIMars(
        ...     hdr_path="data/sample.hdr", annotations="data/labels.mat"
        ... )
        >>> img_data, ann_data = hsi.data()
        >>> print(
        ...     f"Image: {img_data.shape}, Annotations: {ann_data.shape if ann_data else 'None'}"
        ... )

        Notes
        -----
        This method is equivalent to calling `get_img()` and `get_annotations()`
        separately, but provides a more convenient interface when both datasets
        are needed.
        """
        return self.get_img(), self.get_annotations()

    def _prepare_img(self, img: NDArray) -> NDArray:
        """
        Prepare HSI data for visualization by creating a false-color RGB image.

        Selects three spectral bands from the full hyperspectral cube and
        normalizes them to 8-bit RGB format suitable for display.

        Parameters
        ----------
        img : NDArray
            The full hyperspectral image array.

        Returns
        -------
        NDArray
            8-bit RGB image suitable for display with OpenCV.

        Notes
        -----
        Uses the default bands specified in the ENVI header for false-color
        visualization. These typically correspond to visible and near-infrared
        wavelengths that provide good visual contrast.
        """
        # Select the false-color bands for RGB visualization
        img = img[..., self._false_colour_bands]

        # Normalize to 8-bit range [0, 255] for display
        img = cv2.normalize(
            img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return img

    def _prepare_ann(self, img: NDArray) -> NDArray:
        """
        Prepare annotation data for visualization with color mapping.

        Converts label values to an 8-bit format, applies a colormap for
        visual distinction between classes, and preserves background pixels
        as black.

        Parameters
        ----------
        img : NDArray
            The annotation label array.

        Returns
        -------
        NDArray
            8-bit RGB image with colormap applied, suitable for display.

        Notes
        -----
        Pixels with label value 0 (background/unlabeled) are displayed as
        black (0, 0, 0). Other labels are mapped to colors using the TURBO
        colormap for maximum visual distinction.
        """
        # Identify background pixels (label == 0)
        mask = img == 0

        # Normalize labels to 8-bit range
        img = cv2.normalize(
            img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply TURBO colormap for visual distinction between classes
        img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

        # Preserve background as black
        img[mask] = (0, 0, 0)

        return img

    def _cv2_imshow(self, title: str, mat: NDArray) -> None:
        """
        Display an image in an OpenCV window with keyboard interaction.

        Creates a resizable window, displays the image, and waits for a
        keypress before closing the window.

        Parameters
        ----------
        title : str
            Window title.
        mat : NDArray
            Image array to display.

        Notes
        -----
        The window is created with WINDOW_NORMAL flag, making it resizable.
        Press any key to close the window.
        """
        # Create a resizable named window
        cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(title, mat)
        cv2.waitKey(0)  # Wait for any keypress
        cv2.destroyWindow(title)

    def display_hsi(self) -> None:
        """
        Display the hyperspectral image as a false-color RGB visualization.

        Opens an interactive OpenCV window showing the HSI data rendered using
        three representative spectral bands. The window can be resized and
        closed by pressing any key.

        Examples
        --------
        >>> hsi = HSIMars(hdr_path="data/sample.hdr")
        >>> hsi.display_hsi()  # Opens window, press any key to close

        Notes
        -----
        The false-color bands are automatically selected from the ENVI header
        metadata, typically representing visible and near-infrared wavelengths
        for optimal visual interpretation.
        """
        # Load and prepare the HSI for visualization
        img = self.get_img().hsi
        img = self._prepare_img(img)
        self._cv2_imshow(self.hdr_path.stem, img)

    def display_annotations(self) -> None:
        """
        Display the ground truth annotations with color-coded labels.

        Opens an interactive OpenCV window showing the annotation labels
        with a colormap applied for visual distinction between classes.
        Background pixels (label 0) are displayed in black.

        Raises
        ------
        AttributeError
            If no annotations were provided during initialization.

        Examples
        --------
        >>> hsi = HSIMars(
        ...     hdr_path="data/sample.hdr", annotations="data/labels.mat"
        ... )
        >>> hsi.display_annotations()  # Opens window, press any key to close

        Notes
        -----
        The TURBO colormap is applied to provide maximum visual distinction
        between different material classes in the annotations.
        """
        # Load and prepare annotations for visualization
        ann_data = self.get_annotations()
        if ann_data is None:
            raise ValueError(
                "No annotations available. Provide annotations path during initialization."
            )

        ann = ann_data.labels.astype("float32")
        ann = self._prepare_ann(ann)
        self._cv2_imshow(self.hdr_path.stem + " - annotations", ann)

    def display(self) -> None:
        """
        Display comprehensive visualization with HSI, annotations, and overlay.

        Opens an interactive OpenCV window showing:
        - Left panel: False-color HSI visualization
        - Middle panel: Color-coded annotations (if available)
        - Right panel: Semi-transparent overlay of annotations on HSI (if available)

        If no annotations are provided, displays only the HSI.

        Examples
        --------
        >>> # With annotations
        >>> hsi = HSIMars(
        ...     hdr_path="data/sample.hdr", annotations="data/labels.mat"
        ... )
        >>> hsi.display()  # Shows three-panel view

        >>> # Without annotations
        >>> hsi = HSIMars(hdr_path="data/sample.hdr")
        >>> hsi.display()  # Shows only HSI

        Notes
        -----
        The overlay uses 75% weight for the HSI and 25% weight for the
        annotations, providing a good balance between seeing spectral features
        and class boundaries.
        """
        # Load and prepare the HSI
        img = self.get_img().hsi
        img = self._prepare_img(img)

        # If annotations are available, create a three-panel display
        if self.annotations is not None:
            ann_data = self.get_annotations()
            if ann_data is not None:
                ann = ann_data.labels.astype("float32")
                ann = self._prepare_ann(ann)
                # Create semi-transparent overlay (75% HSI, 25% annotations)
                sup = cv2.addWeighted(img, 0.75, ann, 0.25, 0.0)
                # Concatenate horizontally: HSI | Annotations | Overlay
                img = cv2.hconcat([img, ann, sup])

        self._cv2_imshow(self.hdr_path.stem, img)

    def plot_spectra(
        self,
        px: list[int, int] | list[list[int, int]] | NDArray,
        convex_hull: bool = False,
        bands: bool = False,
        output: str | Path | None = None,
    ) -> None:
        """
        Plot spectral signature(s) for specified pixel location(s).

        Generates a line plot showing reflectance/intensity as a function of
        wavelength. For multiple pixels, plots the mean spectrum with standard
        deviation shading. Optionally applies convex hull removal to normalize
        continuum and highlights spectral band regions.

        Parameters
        ----------
        px : list[int, int] | list[list[int, int]] | NDArray
            Pixel coordinates to extract spectra from. Can be:

            - Single pixel: [row, col] or [[row, col]]
            - Multiple pixels: [[row1, col1], [row2, col2], ...] or 2D array

            Coordinates are in (row, column) format, 0-indexed.
        convex_hull : bool, optional
            If True, applies convex hull removal to normalize the spectrum
            continuum. This technique is useful for analyzing absorption
            features by removing the overall spectral shape. Default is False.
        bands : bool, optional
            If True, overlays colored regions indicating spectral band types:
            - VIS (Visible): < 750 nm (green)
            - NIR (Near-Infrared): 750-1400 nm (red)
            - SWIR (Short-Wave Infrared): 1400-3000 nm (blue)
            - MWIR (Mid-Wave Infrared): > 3000 nm (magenta)
            Default is False.
        output : str | Path, optional
            Path to save the plot as an image file. If None (default),
            displays the plot interactively using matplotlib's show().
            The directory will be created if it doesn't exist.

        Examples
        --------
        >>> hsi = HSIMars(hdr_path="data/sample.hdr")
        >>> # Plot single pixel spectrum
        >>> hsi.plot_spectra(px=[100, 200])

        >>> # Plot average spectrum of multiple pixels with standard deviation
        >>> pixels = [[100, 200], [101, 200], [100, 201], [101, 201]]
        >>> hsi.plot_spectra(px=pixels, convex_hull=True, bands=True)

        >>> # Save plot to file
        >>> hsi.plot_spectra(px=[100, 200], output="plots/spectrum.png")

        Notes
        -----
        Convex hull removal is performed using the pysptools library. This
        technique divides the spectrum by its convex hull envelope, effectively
        normalizing the continuum and emphasizing absorption features.

        For multiple pixels, the standard deviation is shown as a shaded region
        around the mean spectrum, providing visual indication of spectral
        variability within the selected region.
        """
        # Load the hyperspectral image data
        img = self.get_img()
        hsi = img.hsi
        wl = img.wavelength

        # Convert pixel coordinates to 2D array format for consistent handling
        px = np.atleast_2d(np.array(px))

        # Extract spectra for the specified pixels
        spec = hsi[px[:, 0], px[:, 1]]

        # Calculate mean and standard deviation
        if spec.shape[0] == 1:
            # Single pixel: no standard deviation
            spec_mean = spec[0]
            spec_std = None
        else:
            # Multiple pixels: compute statistics
            spec_mean = spec.mean(axis=0)
            spec_std = spec.std(axis=0)

        # Apply convex hull removal for continuum normalization if requested
        if convex_hull:
            spec_mean, x_hull, y_hull = convex_hull_removal(spec_mean, wl)
            spec_mean = 1.0 - np.array(spec_mean)

            # Also normalize the standard deviation using the convex hull
            if spec_std is not None:
                # Interpolate hull values at all wavelengths
                interp = splrep(x_hull, y_hull, k=1)
                y_hull = splev(wl, interp, der=0)
                spec_std /= y_hull

        # Create the plot with appropriate size and layout
        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.grid(True, alpha=0.15, ls="dashed")

        # Overlay spectral band regions if requested
        if bands:
            # Position text labels at 75% of the y-axis range
            y_pos = 0.75 * (spec_mean.max() - spec_mean.min()) + spec_mean.min()

            # Visible band (VIS): < 750 nm
            ax.axvspan(wl.min(), 750, color="g", alpha=0.15)
            ax.text(
                wl.min() + (750 - wl.min()) / 2.0,
                y_pos,
                "VIS",
                color="g",
                rotation=90,
                va="bottom",
                ha="center",
            )

            # Near-Infrared band (NIR): 750-1400 nm
            ax.axvspan(750, 1400, color="r", alpha=0.15)
            ax.text(
                750 + (1400 - 750) / 2.0,
                y_pos,
                "NIR",
                color="r",
                rotation=90,
                va="bottom",
                ha="center",
            )

            # Short-Wave Infrared band (SWIR): 1400-3000 nm
            ax.axvspan(1400, 3000, color="b", alpha=0.15)
            ax.text(
                1400 + (3000 - 1400) / 2.0,
                y_pos,
                "SWIR",
                color="b",
                rotation=90,
                va="bottom",
                ha="center",
            )

            # Mid-Wave Infrared band (MWIR): > 3000 nm
            ax.axvspan(3000, wl.max(), color="m", alpha=0.15)
            ax.text(
                3000 + (wl.max() - 3000) / 2.0,
                y_pos,
                "MWIR",
                color="m",
                rotation=90,
                va="bottom",
                ha="center",
            )

        # Plot the mean spectrum
        ax.plot(wl, spec_mean, "k-", label="spectrum")

        # Add standard deviation shading for multiple pixels
        if spec_std is not None:
            ax.fill_between(
                wl,
                (spec_mean - spec_std).clip(0, None),  # Clip to non-negative
                spec_mean + spec_std,
                label="$\\pm \\sigma$",
                color="k",
                alpha=0.15,
            )

            # Add legend when showing standard deviation
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                frameon=False,
            )

        # Set axis labels
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("intensity (a.u.)")

        # Display or save the plot
        if output is None:
            plt.show()
        else:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output))
            plt.close()  # Close the figure to free memory

    def plot_histogram(
        self, band: int | float, output: str | Path | None = None
    ) -> None:
        """
        Plot the intensity distribution histogram for a specific spectral band.

        Generates a probability density histogram showing the distribution of
        pixel intensity values across the image for the selected wavelength band.
        Useful for analyzing the statistical properties of spectral features.

        Parameters
        ----------
        band : int | float
            Spectral band selector. Can be:

            - **int**: Direct band index (0-based) in the spectral dimension.
            - **float**: Wavelength in nanometers. The closest available
              wavelength will be automatically selected.
        output : str | Path, optional
            Path to save the histogram plot as an image file. If None (default),
            displays the plot interactively using matplotlib's show().
            The directory will be created if it doesn't exist.

        Examples
        --------
        >>> hsi = HSIMars(hdr_path="data/sample.hdr")
        >>> # Plot histogram for band at index 100
        >>> hsi.plot_histogram(band=100)

        >>> # Plot histogram for band nearest to 1500 nm wavelength
        >>> hsi.plot_histogram(band=1500.0)

        >>> # Save histogram to file
        >>> hsi.plot_histogram(band=1500.0, output="plots/histogram_1500nm.png")

        Notes
        -----
        The histogram uses 100 bins and is normalized to show probability
        density rather than raw counts. This normalization facilitates
        comparison between different bands or images.

        The histogram includes all valid pixels in the image for the selected
        band, providing a global view of intensity distribution.
        """
        # Load the hyperspectral image data
        img = self.get_img()
        hsi = img.hsi
        wl = img.wavelength

        # Select the appropriate band index
        if isinstance(band, int):
            # Direct index selection
            idx = band
        else:
            # Find closest wavelength to the requested value
            idx = np.argmin(np.abs(wl - band))

        # Extract pixel values for the selected band and flatten to 1D
        hist = hsi[..., idx].ravel()
        actual_wl = wl[idx]

        # Create the histogram plot
        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.grid(True, alpha=0.15, ls="dashed")

        # Plot normalized histogram (probability density)
        ax.hist(hist, bins=100, density=True, histtype="step", align="mid")

        # Set axis labels with wavelength information
        ax.set_xlabel(f"pixel values (band: {actual_wl:.2f} nm)")
        ax.set_ylabel("density")

        # Display or save the plot
        if output is None:
            plt.show()
        else:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output))
            plt.close()  # Close the figure to free memory
