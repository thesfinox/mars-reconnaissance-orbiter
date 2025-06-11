"""
Open and manipulate hyperspectral images (HSI).

Author: Riccardo Finotello <riccardo.finotello@cea.fr>
"""

from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pysptools.spectro import convex_hull_removal
from scipy.interpolate import splev, splrep
from scipy.io import loadmat
from spectral.io import envi

plt.style.use("grayscale")
mpl.rc("font", size=16)


class HSIMars:
    """
    Open HSIs of the Martian soil and their annotations (if available).
    """

    def __init__(
        self,
        hdr_path: str | Path,
        annotations: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        hdr_path : str | Path
            Path to the header (`.hdr` extension) file containing the information.
        annotations : str | Path, optional
            Path to the ground truth file.

        .. warning::

            The `.img` file containing the hyperspectral data must be in the same directory as the corresponding header file.

        Raises
        ------
        FileNotFoundError
            If the header file or the annotations could not be found in the filesystem.
        """
        self.hdr_path: Path = Path(hdr_path)
        if not self.hdr_path.exists():
            raise FileNotFoundError(
                f"The header file {self.hdr_path} could not be found in the filesystem."
            )
        self.annotations: str | Path | None = annotations
        if self.annotations is not None:
            self.annotations = Path(self.annotations)
            if not self.annotations.exists():
                raise FileNotFoundError(
                    f"The annotation file {self.annotations} could not be found in the filesystem."
                )

        # Cache large objects
        self._raw = None
        self._img = None
        self._ann = None

    def get_raw(self) -> envi.BsqFile:
        """
        Recover the RAW hyperspectral data.
        """
        if self._raw is None:
            self._raw = envi.open(self.hdr_path)
        return self._raw

    def get_img(self) -> NamedTuple:
        """
        Recover the image as array, and its metadata.

        Returns
        -------
        NamedTuple
            A collection containing the following attributes:

            - `hsi`: the HSI data,
            - `wavelength`: the list of wavelengths (units: nm),
            - `shape`: the shape of the image,
            - `height`: the height of the image,
            - `width`: the width of the image,
            - `channels`: the number of channels in the image,
            - `dtype`: the data type of the HSI.
        """
        if self._img is not None:
            return self._img

        # Get the raw data
        img = self.get_raw()

        # Recover the image and the metadata
        img_memmap = img.open_memmap()
        img_metadata = img.metadata

        wl = np.array(img_metadata["wavelength"]).astype("float32")
        ignore_value = float(img_metadata["data ignore value"])

        # Select the default bands for visualisation
        bands = np.array(img_metadata["default bands"]).astype(int)
        bands = wl[bands]

        # Find bad bands
        bbl = np.array(img_metadata["bbl"]).astype("bool")
        img_memmap = img_memmap[..., bbl]
        wl = wl[bbl]

        # Crop the image removing the ignore value
        mask = img_memmap != ignore_value

        mask_channels = mask.sum(axis=2)
        good_rows = mask_channels.sum(axis=1) > 0
        good_cols = mask_channels.sum(axis=0) > 0
        img_memmap = img_memmap[good_rows, :][:, good_cols]

        # Find remaining bad channgels
        mask = img_memmap == ignore_value
        idx = np.unique(np.argwhere(mask)[..., 2])
        ch = np.ones((img_memmap.shape[2],), dtype="bool")
        ch[idx] = False
        img_memmap = img_memmap[..., ch]
        wl = wl[ch]

        # Update the default bands
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
        # Get target shape
        img = self.get_img()
        H, W = img.height, img.width
        h, w = mat.shape

        # Pad height and width
        pad_top = (H - h) // 2
        pad_bottom = H - h - pad_top
        pad_left = (W - w) // 2
        pad_right = W - w - pad_left

        return np.pad(mat, ((pad_top, pad_bottom), (pad_left, pad_right)))

    def get_annotations(self) -> NamedTuple:
        """
        Recover the annotations as array.

        Returns
        -------
        NamedTuple
            A collection containing the following attributes:

            - `labels`: the annotation data,
            - `shape`: the shape of the image,
            - `height`: the height of the image,
            - `width`: the width of the image,
            - `dtype`: the data type of the annotation.
        """
        if self._ann is not None:
            return self._ann

        if self.annotations is None:
            return

        # Get the matrix
        ann = loadmat(self.annotations)
        for k, v in ann.items():
            if isinstance(v, np.ndarray):
                mat = v.astype("uint8")

        # Pad as input image
        mat = self._pad_annotations(mat)

        output = namedtuple(
            "HSIMarsAnnotationData",
            ["labels", "shape", "height", "width", "dtype"],
        )
        self._ann = output(
            labels=mat,
            shape=mat.shape,
            height=mat.shape[0],
            width=mat.shape[1],
            dtype="uint8",
        )
        return self._ann

    def data(self):
        """
        Return a complete data object.

        Returns
        -------
        tuple[NamedTuple, NamedTuple]
            Two collections containing the following attributes:

            - **HSIMarsImageData**:
                - `hsi`: the HSI data,
                - `wavelength`: the list of wavelengths (units: nm),
                - `shape`: the shape of the image,
                - `height`: the height of the image,
                - `width`: the width of the image,
                - `channels`: the number of channels in the image,
                - `dtype`: the data type of the HSI.
            - **HSIMarsAnnotationData**:
                - `labels`: the annotation data,
                - `shape`: the shape of the image,
                - `height`: the height of the image,
                - `width`: the width of the image,
                - `dtype`: the data type of the annotation.
        """
        return self.get_img(), self.get_annotations()

    def _prepare_img(self, img: NDArray) -> NDArray:
        # Get the image
        img = img[..., self._false_colour_bands]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Normalize the image
        img = cv2.normalize(
            img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return img

    def _prepare_ann(self, img: NDArray) -> NDArray:
        # Get the image
        mask = img == 0
        img = cv2.normalize(
            img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply a colour map
        img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        img[mask] = (0, 0, 0)

        return img

    def _cv2_imshow(self, title: str, mat: NDArray):
        # Prepare a new named window
        cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(title, mat)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def display_hsi(self):
        """Display the HSI (fake colours)."""

        # Open the image in a new window
        img = self.get_img().hsi
        img = self._prepare_img(img)
        self._cv2_imshow(self.hdr_path.stem, img)

    def display_annotations(self):
        """Display the annotations."""

        # Open the image in a new window
        ann = self.get_annotations().labels.astype("float32")
        ann = self._prepare_ann(ann)
        self._cv2_imshow(self.hdr_path.stem + " - annotations", ann)

    def display(self):
        """Display the full information."""

        # Get the image
        img = self.get_img().hsi
        img = self._prepare_img(img)

        if self.annotations is not None:
            ann = self.get_annotations().labels.astype("float32")
            ann = self._prepare_ann(ann)
            sup = cv2.addWeighted(img, 0.75, ann, 0.25, 0.0)
            img = cv2.hconcat([img, ann, sup])

        self._cv2_imshow(self.hdr_path.stem, img)

    def plot_spectra(
        self,
        px: list[int, int] | list[list[int, int]] | NDArray,
        convex_hull: bool = False,
        bands: bool = False,
        output: str | Path | None = None,
    ):
        """
        Plot the spectrum of a pixel or the average spectrum of a set of pixels.

        Parameters
        ----------
        px : list[int, int] | list[list[int, int]] | NDArray
            The tuple containing the location of the pixel, or the set of pixels to consider.
        convex_hull : bool
            Remove the convex hull of the spectrum. By default `False`.
        bands : bool
            Visualise the IR bands on the plot. By default `False`.
        output : str | Path, optional
            The name of the output file. If `None`, then display the result.
        """
        # Get the image
        img = self.get_img()
        hsi = img.hsi
        wl = img.wavelength

        # Select the pixels
        px = np.atleast_2d(np.array(px))
        spec = hsi[px[:, 0], px[:, 1]]

        if spec.shape[0] == 1:
            spec_mean = spec[0]
            spec_std = None
        else:
            spec_mean = spec.mean(axis=0)
            spec_std = spec.std(axis=0)

        # Convex hull
        if convex_hull:
            spec_mean, x_hull, y_hull = convex_hull_removal(spec_mean, wl)
            spec_mean = 1.0 - np.array(spec_mean)

            # Interpolate the convex hull
            if spec_std is not None:
                interp = splrep(x_hull, y_hull, k=1)
                y_hull = splev(wl, interp, der=0)
                spec_std /= y_hull

        # Plot the spectra
        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.grid(True, alpha=0.15, ls="dashed")

        if bands:
            y_pos = 0.75 * (spec_mean.max() - spec_mean.min()) + spec_mean.min()

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

        ax.plot(wl, spec_mean, "k-", label="spectrum")
        if spec_std is not None:
            ax.fill_between(
                wl,
                (spec_mean - spec_std).clip(0, None),
                spec_mean + spec_std,
                label="$\\pm \\sigma$",
                color="k",
                alpha=0.15,
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                frameon=False,
            )

        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("intensity (a.u.)")

        if output is None:
            plt.show()
        else:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output))

    def plot_histogram(
        self, band: int | float, output: str | Path | None = None
    ):
        """
        Plot the histogram of a particular absorption band.

        Parameters
        ----------
        band : int | float
            The band to consider. If integer, than the particular index will be consider.
        output : str | Path, optional
            The name of the output file. If `None`, then display the result.
        """
        # Get the image
        img = self.get_img()
        hsi = img.hsi
        wl = img.wavelength
        idx = band if isinstance(band, int) else np.argmin(np.abs(wl - band))
        hist = hsi[..., idx].ravel()
        wl = wl[idx]

        # Plot the histogram
        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.grid(True, alpha=0.15, ls="dashed")

        ax.hist(hist, bins=100, density=True, histtype="step", align="mid")

        ax.set_xlabel(f"pixel values (band: {wl:.2f} nm)")
        ax.set_ylabel("density")

        if output is None:
            plt.show()
        else:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output))
