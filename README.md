# SpatialWarp

Technology-agnostic alignment of spatial datasets: grid self-alignment (for
technologies whose spot coordinates don't natively match their own reference
image), landmark-guided elastic cross-modality registration, and optional
fine-tuning of a registration using matched-feature correlation.

**Start here:** [`examples/tutorial.ipynb`](examples/tutorial.ipynb) walks through the whole
package end-to-end on synthetic data (three Gaussian blobs standing in for tissue), so it runs
with no external files and no real dataset needed. It builds a fake MSI section and a fake
Visium section, runs grid self-alignment, then cross-modality registration, then `sw.align()`,
with a QC plot after each step to show what "correct" looks like — and a toggle
(`INTERACTIVE = True/False`) to either click landmarks yourself in the real interactive tools or
run the whole thing unattended using the synthetic data's known ground truth. For a real
dataset, go from there to [`examples/l12_walkthrough.ipynb`](examples/l12_walkthrough.ipynb).

From there, `examples/` covers several real use cases built on the same primitives:

- **MSI onto Visium** — [`l12_walkthrough.ipynb`](examples/l12_walkthrough.ipynb) (MSI's own
  H&E aligned first, then registered to Visium HD), or
  [`l12_direct_to_visium.ipynb`](examples/l12_direct_to_visium.ipynb) if the MSI grid already
  matches Visium's H&E directly, or
  [`msi_to_classic_visium.ipynb`](examples/msi_to_classic_visium.ipynb) for classic
  (non-HD) Visium.
- **Multiple MSI modalities on one slide** —
  [`three_msi_modalities_visium_hd.ipynb`](examples/three_msi_modalities_visium_hd.ipynb) aligns
  metabolomics, lipidomics-negative, and lipidomics-positive (three different physical sections,
  three separate registrations) onto a single Visium HD slide, then merges all three into one
  object; [`transfer_msi_to_finer_resolutions.ipynb`](examples/transfer_msi_to_finer_resolutions.ipynb)
  reuses those saved registrations to transfer the same data onto finer bin sizes and cell
  segmentation without re-registering.
- **Serial Visium sections (no MSI)** —
  [`align_two_visium_sections.ipynb`](examples/align_two_visium_sections.ipynb) and
  [`align_five_visium_sections.ipynb`](examples/align_five_visium_sections.ipynb) register
  Visium sections directly to each other (or to a common reference), the same `sw.align()`
  pipeline with both sides being Visium instead of one side being MSI.
- **Multiplex immunofluorescence onto Visium HD** —
  [`multiplex_visium_hd.ipynb`](examples/multiplex_visium_hd.ipynb): an 8-channel IF scan needs
  no grid self-alignment (every channel is already a pixel-perfect raster of itself), just
  cross-modality registration plus the fine-tuning step.

## Three building blocks

- **Grid self-alignment** (`spatialwarp.grid_align.run_grid_alignment` /
  `align_grid`): some spatial technologies (e.g. MSI instrument rasters)
  report spot coordinates in a system that doesn't share an origin/scale/
  rotation with their own reference image. This interactive slider tool
  fixes that before anything cross-modality happens. Not every technology
  needs it — a multiplex IF scan or Visium's own bins are already
  pixel-perfect rasters of themselves, so this step is skipped for those.
- **Cross-modality registration** (`spatialwarp.pipeline.align`): aligning
  two *different* images (e.g. an MSI section's H&E and a Visium slide's
  H&E) via landmark-guided elastic (B-spline) registration through
  SimpleITK, then warping one dataset's points into the other's space and
  nearest-neighbor matching them.
- **Registration fine-tuning** (`spatialwarp.refine.refine_alignment` /
  `refine_alignment_elastic`): once features have been transferred across,
  you often already know a marker that *should* correspond between the two
  modalities (a gene and the protein it encodes, say). These nudge an
  existing registration — a small rigid correction, or a local B-spline one
  — to maximize that correlation directly, instead of re-picking landmarks.
  Both include a safety net: the fit has to beat a no-correction baseline by
  a minimum margin (`min_improvement`) or the original registration is
  returned unchanged, since a weak/noisy correlation signal can easily make
  a search converge somewhere worse than doing nothing.

`spatialwarp.pipeline.align()` operates on two `spatialdata.SpatialData`
objects — one image element, one table element with `obsm['spatial']` — so
any technology works once your data is in that shape.
`spatialwarp.spatialdata_builder.build_spatialdata()` packages a raw
`(image, points_xy, values)` triple into that shape; `examples/msi/` shows
one concrete loader (MSI vendor exports) that produces it.

## Package layout

| Module | What it does |
|---|---|
| `grid_align` | Interactive slider tool for a technology's own grid-to-image alignment |
| `landmark_picker` | Click corresponding points between two images (or point-cloud grids) |
| `registration` | `register_elastic()` + `RegistrationResult` (SimpleITK affine + B-spline) |
| `refine` | Fine-tune an existing `RegistrationResult` from matched-feature correlation |
| `pipeline` | `align()` — orchestrates registration, point warping, and nearest-neighbor matching |
| `matching` | Plain `cKDTree` nearest-neighbor matching between two point sets |
| `raster` | Rasterize a point cloud onto its own (or a shared) native pixel grid |
| `spatialdata_builder` | Pack/unpack `(image, points, values)` into/out of a `SpatialData` object |
| `qc` | Interactive overlay + alpha slider to visually check an alignment |
| `bunwarp` | Read BUnwarpJ transform files (Fiji), for the legacy manual-registration path |

## Install

```bash
pip install -e .            # core
pip install -e ".[examples]"  # + example notebook dependencies (geopandas, spatialdata-io, ...)
```

## Usage

```python
import spatialwarp as sw

# Click a handful of corresponding points between the two H&E images.
moving_landmarks, fixed_landmarks, _, _ = sw.pick_landmarks(moving_he_array, fixed_he_array)

registration_result = sw.register_elastic(
    moving_image=moving_he_array,
    fixed_image=fixed_he_array,
    moving_landmarks=moving_landmarks,
    fixed_landmarks=fixed_landmarks,
)

merged = sw.align(
    moving=my_msi_sdata,      # spatialdata.SpatialData
    fixed=my_visium_sdata,    # spatialdata.SpatialData
    registration_result=registration_result,
    distance_threshold=20.0,
)
```

Registration only depends on the two H&E images, not the analyte, so the same
`registration_result` can be reused across MSI modalities (e.g. metabolomics
and lipidomics on the same slide) without re-registering.
`RegistrationResult.save()`/`.load()` persist it to disk — including any
fine-tuning correction, saved alongside as an extra `.correction.tfm`.

Optionally, tighten the registration using a marker you already know should
correspond between the two modalities:

```python
merged, refined_registration_result = sw.refine_alignment(
    moving=my_msi_sdata,
    fixed=my_visium_sdata,
    registration_result=registration_result,
    feature_pairs=[("aSMA_780 (Opal 780)", "ACTA2")],  # (moving feature, fixed feature)
    distance_threshold=20.0,
)
```

## Examples

`examples/` has one notebook per scenario, all built on the same primitives above:

| Notebook | Demonstrates |
|---|---|
| `tutorial.ipynb` | Minimal end-to-end walkthrough |
| `l12_walkthrough.ipynb` | Full worked example: MSI metabolomics + lipidomics onto Visium HD |
| `l12_direct_to_visium.ipynb` | Aligning MSI directly onto Visium HD (skipping the own-H&E step) |
| `msi_to_classic_visium.ipynb` | Aligning MSI onto classic Visium (Space Ranger, not Visium HD) |
| `three_msi_modalities_visium_hd.ipynb` | Three MSI modalities (metabolomics, lipidomics pos/neg), each its own section, onto one Visium HD slide |
| `three_msi_modalities_figures.ipynb` | Plotting each MSI modality in its own native (unaligned) coordinates |
| `transfer_msi_to_finer_resolutions.ipynb` | Reusing a saved registration to transfer data onto finer bin sizes and cell segmentation, without re-registering |
| `multiplex_visium_hd.ipynb` | Multiplex immunofluorescence (8-channel) onto Visium HD, plus registration fine-tuning |
| `align_two_visium_sections.ipynb` | Aligning two Visium sections to each other |
| `align_five_visium_sections.ipynb` | Aligning five Visium sections to a common reference |
| `enhanced_msi_visium_analysis.ipynb` | Downstream analysis of a BayesSpace-enhanced, SpatialWarp-aligned object |

None of the data these notebooks read/write is tracked in this repo (see
`.gitignore`) — point their file paths at your own data to run them.

## Scope

The core pipeline stops at the merged/matched `AnnData` — clustering,
cell-type scoring, correlation heatmaps, and other downstream analysis are
out of scope for the package itself and vary per project (though several
example notebooks explore this further, for illustration).
