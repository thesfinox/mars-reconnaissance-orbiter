Advanced Examples
=================

This page provides advanced usage examples for the HSI Mars package.

Batch Processing Multiple Images
---------------------------------

Process multiple hyperspectral images in a loop:

.. code-block:: python

   from pathlib import Path
   from hsimars import HSIMars

   # Directory containing CRISM data
   data_dir = Path("data/crism_images")
   output_dir = Path("results/batch_analysis")
   output_dir.mkdir(parents=True, exist_ok=True)

   # Find all .hdr files
   hdr_files = list(data_dir.glob("*.hdr"))

   print(f"Processing {len(hdr_files)} images...")

   for hdr_path in hdr_files:
       print(f"\nProcessing: {hdr_path.name}")

       # Load image
       hsi = HSIMars(hdr_path=hdr_path)
       img_data = hsi.get_img()

       # Save spectrum from center pixel
       center = [img_data.height // 2, img_data.width // 2]
       output_file = output_dir / f"{hdr_path.stem}_spectrum.png"

       hsi.plot_spectra(
           px=center,
           convex_hull=True,
           bands=True,
           output=output_file
       )

       print(f"  - Saved spectrum to {output_file}")

Region of Interest Analysis
----------------------------

Analyze a specific region of interest within an image:

.. code-block:: python

   import numpy as np
   from hsimars import HSIMars

   # Load image
   hsi = HSIMars(hdr_path="data/sample.hdr")
   img_data = hsi.get_img()

   # Define region of interest (ROI)
   roi_top_left = [100, 150]
   roi_bottom_right = [200, 250]

   # Extract all pixels in ROI
   roi_pixels = []
   for r in range(roi_top_left[0], roi_bottom_right[0] + 1):
       for c in range(roi_top_left[1], roi_bottom_right[1] + 1):
           roi_pixels.append([r, c])

   # Plot average spectrum of ROI with standard deviation
   hsi.plot_spectra(
       px=roi_pixels,
       convex_hull=True,
       bands=True,
       output="results/roi_spectrum.png"
   )

   # Calculate statistics
   roi_spectra = img_data.hsi[
       roi_top_left[0]:roi_bottom_right[0]+1,
       roi_top_left[1]:roi_bottom_right[1]+1,
       :
   ]

   print(f"ROI dimensions: {roi_spectra.shape[:2]}")
   print(f"ROI pixels: {roi_spectra.shape[0] * roi_spectra.shape[1]}")
   print(f"Mean intensity: {roi_spectra.mean():.2f}")
   print(f"Std intensity: {roi_spectra.std():.2f}")

Class-Based Spectral Analysis
------------------------------

Analyze spectral signatures for different annotated classes:

.. code-block:: python

   import numpy as np
   from hsimars import HSIMars

   # Load data with annotations
   hsi = HSIMars(
       hdr_path="data/sample.hdr",
       annotations="data/sample_labels.mat"
   )

   img_data, ann_data = hsi.data()

   # Get unique class labels (excluding background = 0)
   classes = np.unique(ann_data.labels)
   classes = classes[classes > 0]

   print(f"Found {len(classes)} classes")

   # Analyze each class
   for class_id in classes:
       # Find all pixels belonging to this class
       class_mask = ann_data.labels == class_id
       class_coords = np.argwhere(class_mask)

       # Convert to list of [row, col] pairs
       class_pixels = class_coords.tolist()

       print(f"\nClass {class_id}: {len(class_pixels)} pixels")

       # Plot average spectrum for this class
       if len(class_pixels) > 0:
           hsi.plot_spectra(
               px=class_pixels,
               convex_hull=True,
               bands=True,
               output=f"results/class_{class_id}_spectrum.png"
           )

Spectral Band Comparison
-------------------------

Compare specific spectral bands across the image:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hsimars import HSIMars

   # Load image
   hsi = HSIMars(hdr_path="data/sample.hdr")
   img_data = hsi.get_img()

   # Select wavelengths of interest
   wavelengths_of_interest = [700, 1000, 1500, 2000]  # nm

   # Find closest band indices
   band_indices = [
       np.argmin(np.abs(img_data.wavelength - wl))
       for wl in wavelengths_of_interest
   ]

   # Create comparison plot
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.ravel()

   for idx, (band_idx, wl) in enumerate(zip(band_indices, wavelengths_of_interest)):
       actual_wl = img_data.wavelength[band_idx]
       band_data = img_data.hsi[:, :, band_idx]

       im = axes[idx].imshow(band_data, cmap='gray')
       axes[idx].set_title(f'Band: {actual_wl:.1f} nm (target: {wl} nm)')
       axes[idx].axis('off')
       plt.colorbar(im, ax=axes[idx], fraction=0.046)

   plt.tight_layout()
   plt.savefig('results/band_comparison.png', dpi=150, bbox_inches='tight')
   print("Saved band comparison to results/band_comparison.png")

Statistical Summary Generation
-------------------------------

Generate comprehensive statistics for a dataset:

.. code-block:: python

   import numpy as np
   from hsimars import HSIMars

   def generate_statistics_report(hsi_path, ann_path=None, output_file="report.txt"):
       """Generate a comprehensive statistics report."""

       # Load data
       if ann_path:
           hsi = HSIMars(hdr_path=hsi_path, annotations=ann_path)
           img_data, ann_data = hsi.data()
       else:
           hsi = HSIMars(hdr_path=hsi_path)
           img_data = hsi.get_img()
           ann_data = None

       # Prepare report
       lines = []
       lines.append("="*60)
       lines.append("Hyperspectral Image Statistics Report")
       lines.append("="*60)
       lines.append(f"\nFile: {hsi_path}")
       lines.append(f"\nImage Properties:")
       lines.append(f"  - Dimensions: {img_data.height} x {img_data.width}")
       lines.append(f"  - Spectral bands: {img_data.channels}")
       lines.append(f"  - Data type: {img_data.dtype}")
       lines.append(f"  - Total pixels: {img_data.height * img_data.width:,}")

       lines.append(f"\nSpectral Information:")
       lines.append(f"  - Wavelength range: {img_data.wavelength.min():.2f} - {img_data.wavelength.max():.2f} nm")
       lines.append(f"  - Wavelength spacing: {np.diff(img_data.wavelength).mean():.2f} nm (mean)")

       lines.append(f"\nIntensity Statistics:")
       lines.append(f"  - Global mean: {img_data.hsi.mean():.4f}")
       lines.append(f"  - Global std: {img_data.hsi.std():.4f}")
       lines.append(f"  - Global min: {img_data.hsi.min():.4f}")
       lines.append(f"  - Global max: {img_data.hsi.max():.4f}")

       # Per-band statistics
       band_means = img_data.hsi.mean(axis=(0, 1))
       band_stds = img_data.hsi.std(axis=(0, 1))

       lines.append(f"\nPer-Band Statistics:")
       lines.append(f"  - Mean intensity range: {band_means.min():.4f} - {band_means.max():.4f}")
       lines.append(f"  - Highest mean at: {img_data.wavelength[band_means.argmax()]:.2f} nm")
       lines.append(f"  - Lowest mean at: {img_data.wavelength[band_means.argmin()]:.2f} nm")

       # Annotation statistics
       if ann_data is not None:
           unique_labels, counts = np.unique(ann_data.labels, return_counts=True)

           lines.append(f"\nAnnotation Statistics:")
           lines.append(f"  - Total classes: {len(unique_labels)}")
           lines.append(f"  - Class distribution:")

           for label, count in zip(unique_labels, counts):
               percentage = (count / ann_data.labels.size) * 100
               label_name = "Background" if label == 0 else f"Class {label}"
               lines.append(f"    * {label_name}: {count:,} pixels ({percentage:.2f}%)")

       lines.append("\n" + "="*60)

       # Write to file
       report_text = "\n".join(lines)
       with open(output_file, 'w') as f:
           f.write(report_text)

       print(report_text)
       print(f"\nReport saved to: {output_file}")

       return report_text

   # Usage
   generate_statistics_report(
       hsi_path="data/sample.hdr",
       ann_path="data/sample_labels.mat",
       output_file="results/statistics_report.txt"
   )

Working with Specific Wavelength Ranges
----------------------------------------

Extract and analyze specific portions of the spectrum:

.. code-block:: python

   import numpy as np
   from hsimars import HSIMars

   # Load image
   hsi = HSIMars(hdr_path="data/sample.hdr")
   img_data = hsi.get_img()

   # Define wavelength range of interest (e.g., SWIR: 1400-3000 nm)
   wl_min, wl_max = 1400, 3000

   # Find bands within range
   mask = (img_data.wavelength >= wl_min) & (img_data.wavelength <= wl_max)
   selected_bands = np.where(mask)[0]
   selected_wavelengths = img_data.wavelength[mask]

   print(f"Selected {len(selected_bands)} bands in range {wl_min}-{wl_max} nm")
   print(f"Wavelength range: {selected_wavelengths.min():.1f} - {selected_wavelengths.max():.1f} nm")

   # Extract subset
   swir_data = img_data.hsi[:, :, selected_bands]

   # Analyze average spectrum in this range
   center = [img_data.height // 2, img_data.width // 2]
   center_spectrum = swir_data[center[0], center[1], :]

   # Plot
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.plot(selected_wavelengths, center_spectrum, 'k-')
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Intensity (a.u.)')
   plt.title(f'SWIR Spectrum ({wl_min}-{wl_max} nm) at Center Pixel')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('results/swir_spectrum.png', dpi=150, bbox_inches='tight')
   print("Saved SWIR spectrum plot")

Custom Visualization
--------------------

Create custom visualizations using matplotlib:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.colors import ListedColormap
   from hsimars import HSIMars

   # Load data
   hsi = HSIMars(
       hdr_path="data/sample.hdr",
       annotations="data/sample_labels.mat"
   )
   img_data, ann_data = hsi.data()

   # Create custom figure
   fig = plt.figure(figsize=(15, 5))

   # Panel 1: False-color image
   ax1 = plt.subplot(131)
   # Use specific bands for RGB
   false_color = img_data.hsi[:, :, [50, 100, 150]]
   false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())
   ax1.imshow(false_color)
   ax1.set_title('False Color Image')
   ax1.axis('off')

   # Panel 2: Annotations
   ax2 = plt.subplot(132)
   # Create custom colormap
   n_classes = len(np.unique(ann_data.labels))
   colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
   cmap = ListedColormap(colors)

   im2 = ax2.imshow(ann_data.labels, cmap=cmap, interpolation='nearest')
   ax2.set_title('Ground Truth Labels')
   ax2.axis('off')
   plt.colorbar(im2, ax=ax2, fraction=0.046)

   # Panel 3: Single band
   ax3 = plt.subplot(133)
   band_idx = img_data.channels // 2  # Middle band
   im3 = ax3.imshow(img_data.hsi[:, :, band_idx], cmap='gray')
   ax3.set_title(f'Band {band_idx} ({img_data.wavelength[band_idx]:.1f} nm)')
   ax3.axis('off')
   plt.colorbar(im3, ax=ax3, fraction=0.046)

   plt.tight_layout()
   plt.savefig('results/custom_visualization.png', dpi=150, bbox_inches='tight')
   print("Saved custom visualization")

Performance Tips
----------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process in chunks to manage memory
   import numpy as np
   from hsimars import HSIMars

   hsi = HSIMars(hdr_path="data/large_image.hdr")
   img_data = hsi.get_img()

   # Process image in tiles
   tile_size = 100
   results = []

   for i in range(0, img_data.height, tile_size):
       for j in range(0, img_data.width, tile_size):
           tile = img_data.hsi[
               i:min(i+tile_size, img_data.height),
               j:min(j+tile_size, img_data.width),
               :
           ]
           # Process tile
           result = tile.mean()  # Example operation
           results.append(result)

   print(f"Processed {len(results)} tiles")

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from multiprocessing import Pool
   from hsimars import HSIMars

   def process_image(hdr_path):
       """Process a single image."""
       hsi = HSIMars(hdr_path=hdr_path)
       img_data = hsi.get_img()

       # Extract some metric
       mean_spectrum = img_data.hsi.mean(axis=(0, 1))

       return {
           'filename': hdr_path.name,
           'mean_intensity': mean_spectrum.mean(),
           'shape': img_data.shape
       }

   # Find all images
   data_dir = Path("data/crism_images")
   hdr_files = list(data_dir.glob("*.hdr"))

   # Process in parallel
   with Pool(processes=4) as pool:
       results = pool.map(process_image, hdr_files)

   # Print summary
   for result in results:
       print(f"{result['filename']}: mean={result['mean_intensity']:.4f}")

Further Resources
-----------------

* See the :doc:`modules/modules` for complete API documentation
* Check the GitHub repository for example notebooks
* Join discussions on GitHub Issues
