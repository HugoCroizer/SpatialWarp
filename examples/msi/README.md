# MSI example integration

This folder shows one way to turn raw MSI (mass spectrometry imaging) vendor
exports into a `spatialdata.SpatialData` object that the generic
`spatialwarp` core package can consume. None of this is part of the core
package — it's specific to this MSI vendor's CSV format and the fact that its
spot grid needs interactive self-alignment before use (see `grid_align.py` in
the core package for why).

- `msi_loader.py` — `MSIdata`: parses the vendor's counts/metabolites/region
  CSVs and maps m/z values to metabolite names.
- `build_msi_spatialdata.py` — `build_msi_spatialdata(...)`: loads an MSI
  dataset with `MSIdata`, runs `spatialwarp.grid_align.run_grid_alignment()`
  to align the MSI grid onto its own H&E image, and packages the result as a
  `SpatialData` object (one image element, one table element with
  `obsm['spatial']`).

To align this against another modality (e.g. Visium, itself loaded into a
`SpatialData` object via `spatialdata_io.visium_hd`), call
`spatialwarp.pipeline.align(moving=msi_sdata, fixed=visium_sdata, ...)`.
See `../l12_walkthrough.ipynb` for a full worked example.
