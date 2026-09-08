# MSI example integration

This folder shows the one MSI-specific piece needed to use `spatialwarp` with
mass spectrometry imaging data: parsing this vendor's CSV export format.
Everything else — placing the MSI grid on its own H&E image, registering that
image against another modality, and packaging the result as a `SpatialData`
object — is done with the core package's generic, technology-agnostic
primitives, composed directly in `../l12_walkthrough.ipynb`:

- `msi_loader.py` — `MSIdata`: parses the vendor's counts/metabolites/region
  CSVs and maps m/z values to metabolite names. This is the only file here;
  it produces plain point coordinates + a feature `DataFrame`, nothing more.

From there, the notebook composes three reusable primitives to place the MSI
grid onto its own H&E image (the same primitives used for cross-modality
registration against Visium, just with a rasterized point cloud on one side
instead of a second real image):

```python
from spatialwarp.landmark_picker import pick_landmarks
from spatialwarp.registration import register_elastic
from spatialwarp.raster import rasterize_points, points_to_pixel
from spatialwarp.spatialdata_builder import build_spatialdata

# 1. landmark
moving_lm, fixed_lm, _, feature = pick_landmarks(he_image, (points_xy, feature_values))

# 2. register
intensity = feature_values[feature] if feature is not None else feature_values.sum(axis=1)
pseudo_image, transform = rasterize_points(points_xy, intensity)
result = register_elastic(moving_image=he_image, fixed_image=pseudo_image,
                           moving_landmarks=moving_lm, fixed_landmarks=fixed_lm)

# 3. transform
pixel_xy = points_to_pixel(points_xy, transform)
aligned_x, aligned_y = result.warp_points_fixed_to_moving(pixel_xy[:, 0], pixel_xy[:, 1])

# package
sdata = build_spatialdata(he_image, np.column_stack([aligned_x, aligned_y]), values=feature_values)
```

Then align this against another modality (e.g. Visium, loaded via
`spatialdata_io.visium_hd`) with `spatialwarp.pipeline.align(moving=sdata, fixed=visium_sdata, ...)`.
See `../l12_walkthrough.ipynb` for the full worked example.
