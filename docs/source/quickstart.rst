Quick Start Guide
=================

This guide will help you get started with the HSI Mars package quickly.

Basic Usage
-----------

Loading a Hyperspectral Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest use case is loading and displaying a hyperspectral image:

.. code-block:: python

   from hsimars import HSIMars

   # Load a hyperspectral image
   hsi = HSIMars(hdr_path="path/to/your/image.hdr")

   # Get image data and metadata
   img_data = hsi.get_img()

   # Print basic information
   print(f"Image shape: {img_data.shape}")
   print(f"Height: {img_data.height} pixels")
   print(f"Width: {img_data.width} pixels")
   print(f"Channels: {img_data.channels} spectral bands")
   print(f"Wavelength range: {img_data.wavelength.min():.1f} - {img_data.wavelength.max():.1f} nm")

.. note::

   The ``.hdr`` file must be accompanied by a corresponding ``.img`` file in the same directory.
   Both files are part of the ENVI format standard used by CRISM.

Displaying Images
~~~~~~~~~~~~~~~~~

Display the false-color visualization:

.. code-block:: python

   # Display the hyperspectral image as false-color RGB
   hsi.display_hsi()

   # This opens an interactive OpenCV window
   # Press any key to close the window

Working with Annotations
------------------------

If you have ground truth annotations for your hyperspectral image:

Loading Annotations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hsimars import HSIMars

   # Load image with annotations
   hsi = HSIMars(
       hdr_path="path/to/image.hdr",
       annotations="path/to/labels.mat"
   )

   # Get both image and annotation data
   img_data, ann_data = hsi.data()

   # Print annotation information
   if ann_data is not None:
       print(f"Annotation shape: {ann_data.shape}")
       print(f"Number of classes: {len(np.unique(ann_data.labels))}")

Visualizing Annotations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Display annotations only
   hsi.display_annotations()

   # Display combined view: image, annotations, and overlay
   hsi.display()
   # This shows three panels side by side:
   # 1. False-color image
   # 2. Color-coded annotations
   # 3. Semi-transparent overlay

Spectral Analysis
-----------------

Plotting Spectra
~~~~~~~~~~~~~~~~

Analyze the spectral signature of specific pixels:

.. code-block:: python

   # Plot spectrum for a single pixel at coordinates (100, 200)
   hsi.plot_spectra(px=[100, 200])

   # Plot average spectrum from multiple pixels
   pixels = [[100, 200], [101, 200], [100, 201], [101, 201]]
   hsi.plot_spectra(px=pixels)

Advanced Spectral Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply convex hull removal and show spectral bands:

.. code-block:: python

   # Plot with convex hull removal (continuum normalization)
   hsi.plot_spectra(
       px=[100, 200],
       convex_hull=True,
       bands=True  # Show VIS, NIR, SWIR, MWIR regions
   )

   # Save the plot to a file
   hsi.plot_spectra(
       px=[100, 200],
       convex_hull=True,
       bands=True,
       output="results/spectrum_100_200.png"
   )

Histogram Analysis
~~~~~~~~~~~~~~~~~~

Generate histograms for specific spectral bands:

.. code-block:: python

   # Histogram for a wavelength (automatically finds closest band)
   hsi.plot_histogram(band=1500.0)  # 1500 nm

   # Histogram by band index
   hsi.plot_histogram(band=100)

   # Save histogram to file
   hsi.plot_histogram(
       band=1500.0,
       output="results/histogram_1500nm.png"
   )

Complete Example
----------------

Here's a complete workflow combining multiple operations:

.. code-block:: python

   import numpy as np
   from hsimars import HSIMars

   # Load data
   hsi = HSIMars(
       hdr_path="data/sample.hdr",
       annotations="data/sample_labels.mat"
   )

   # Get data
   img_data, ann_data = hsi.data()

   # Print summary
   print("="*50)
   print("HSI Data Summary")
   print("="*50)
   print(f"Image dimensions: {img_data.height} x {img_data.width}")
   print(f"Spectral bands: {img_data.channels}")
   print(f"Wavelength range: {img_data.wavelength.min():.1f} - {img_data.wavelength.max():.1f} nm")

   if ann_data is not None:
       unique_labels = np.unique(ann_data.labels)
       print(f"Number of classes: {len(unique_labels)}")
       print(f"Class labels: {unique_labels}")

   # Visualize
   hsi.display()  # Interactive display

   # Analyze specific region
   center_pixel = [img_data.height // 2, img_data.width // 2]
   print(f"\nAnalyzing pixel at {center_pixel}")

   hsi.plot_spectra(
       px=center_pixel,
       convex_hull=True,
       bands=True,
       output="results/center_spectrum.png"
   )

   # Generate histogram for a key wavelength
   hsi.plot_histogram(
       band=1500.0,
       output="results/histogram_1500nm.png"
   )

   print("\nAnalysis complete! Check the 'results/' directory for plots.")

Memory Considerations
---------------------

The HSI Mars package uses lazy loading to manage memory efficiently:

.. code-block:: python

   # Create the object (no data loaded yet)
   hsi = HSIMars(hdr_path="path/to/large_image.hdr")

   # Data is loaded only when first accessed
   img_data = hsi.get_img()  # Loads and caches data

   # Subsequent calls use cached data (no disk I/O)
   img_data2 = hsi.get_img()  # Returns cached data

   # The same applies to annotations
   ann_data = hsi.get_annotations()  # Loads and caches
   ann_data2 = hsi.get_annotations()  # Returns cached data

Best Practices
--------------

1. **Use context-appropriate coordinates**: Remember that pixel coordinates are in ``(row, column)`` format, which corresponds to ``(y, x)`` in image coordinates.

2. **Check for annotations**: Always verify that annotations exist before trying to display them:

   .. code-block:: python

      if hsi.annotations is not None:
          hsi.display_annotations()
      else:
          print("No annotations available")

3. **Close OpenCV windows**: When using display methods, the window stays open until you press a key. This is intentional for interactive exploration.

4. **Save plots programmatically**: Use the ``output`` parameter to save plots instead of displaying them interactively when processing multiple images.

5. **Work with subsets**: For large datasets, consider analyzing specific regions of interest rather than processing the entire image at once.
