# SpatialWarp

Technology-agnostic alignment of two spatial datasets: grid self-alignment (for
technologies whose spot coordinates don't natively match their own reference
image) plus landmark-guided elastic cross-modality registration.
## Two separate problems

- **Grid self-alignment** (`spatialwarp.grid_align.run_grid_alignment`): some
  spatial technologies (e.g. MSI instrument rasters) report spot coordinates
  in a system that doesn't share an origin/scale/rotation with their own
  reference image. This interactive slider tool fixes that before anything
  cross-modality happens.
- **Cross-modality registration** (`spatialwarp.pipeline.align`): aligning two
  *different* images (e.g. an MSI section's H&E and a Visium slide's H&E) via
  landmark-guided elastic (B-spline) registration through SimpleITK, then
  warping one dataset's points into the other's space and nearest-neighbor
  matching them.

`spatialwarp.pipeline.align()` operates on two `spatialdata.SpatialData`
objects — one image element, one table element with `obsm['spatial']` — so
any technology works once your data is in that shape. `examples/msi/` shows
one concrete way to get MSI vendor exports into that shape.

## Install

```bash
pip install -e .            # core
pip install -e ".[examples]"  # + MSI example dependencies
```

## Usage

See `examples/l12_walkthrough.ipynb` for a full worked example (MSI
metabolomics + lipidomics aligned to Visium HD, reproducing the original
per-slide notebook this package replaces).

```python
import spatialwarp
from spatialwarp.landmark_picker import pick_landmarks
from spatialwarp.registration import register_elastic

# Click a handful of corresponding points between the two H&E images.
moving_landmarks, fixed_landmarks = pick_landmarks(moving_he_array, fixed_he_array)

registration_result = register_elastic(
    moving_image=moving_he_array,
    fixed_image=fixed_he_array,
    moving_landmarks=moving_landmarks,
    fixed_landmarks=fixed_landmarks,
)

adata = spatialwarp.align(
    moving=my_msi_sdata,      # spatialdata.SpatialData
    fixed=my_visium_sdata,    # spatialdata.SpatialData
    registration_result=registration_result,
    distance_threshold=20.0,
)
```

Registration only depends on the two H&E images, not the analyte, so the same
`registration_result` can be reused across MSI modalities (e.g. metabolomics
and lipidomics on the same slide) without re-registering.
`RegistrationResult.save()`/`.load()` persist it to disk.

## Scope

The pipeline stops at the merged/matched `AnnData` — downstream analysis
(cell-type scoring, clustering, correlation heatmaps, etc.) is intentionally
out of scope and varies per project.

