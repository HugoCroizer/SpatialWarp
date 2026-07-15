"""Example: turn a raw MSI export into a SpatialData object ready for
``spatialwarp.pipeline.align()``.

This is the concrete instance of "Problem A" from the SpatialWarp design:
MSI spot coordinates don't natively align to the MSI section's own H&E image,
so we run the interactive grid self-alignment tool first.
"""

import anndata
import imageio.v2 as imageio
import numpy as np
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, TableModel

from spatialwarp.grid_align import run_grid_alignment

from msi_loader import MSIdata


def build_msi_spatialdata(
    image_path,
    intensity_csv,
    annotation_csv,
    region_csv,
    x_flip=True,
    aligned_coords_csv=None,
    feature=None,
    cmap="viridis",
):
    """Load an MSI dataset, interactively align its grid onto its own H&E
    image, and package the result as a SpatialData object.

    Parameters
    ----------
    image_path : str
        Path to the MSI section's own H&E TIFF.
    intensity_csv, annotation_csv, region_csv : str
        Vendor MSI export files (see ``MSIdata``).
    x_flip : bool
        Passed to ``run_grid_alignment`` — this vendor's raster x-axis runs
        opposite to the image's, so this defaults to True.
    aligned_coords_csv : str or None
        If given, the aligned coordinates are also saved here.
    feature : str or None
        Which metabolite/lipid column to use as the alignment image. If None,
        an interactive Prev/Next browser lets you pick the one with the
        clearest contrast (see
        :func:`spatialwarp.grid_align.pick_landmarks_multi_feature`); pass a
        specific column name to skip the browser once you already know which
        one works best.
    cmap : str
        Colormap for the feature-browsing picker's pseudo-image.

    Returns
    -------
    spatialdata.SpatialData
        One image element (``"he"``) and one table element (``"msi"``) whose
        ``obsm['spatial']`` holds the aligned pixel coordinates.
    """
    msi = MSIdata(intensity_csv, annotation_csv, region_csv)
    counts = msi.get_count_formated()  # spots x metabolites, plus x/y columns

    image = imageio.imread(image_path)
    points_xy = msi.coordinates[["x", "y"]].values
    feature_values = counts.drop(columns=["x", "y"])

    aligned_xy = run_grid_alignment(
        image=image,
        points_xy=points_xy,
        values=feature_values,
        feature=feature,
        output_csv=aligned_coords_csv,
        x_flip=x_flip,
        cmap=cmap,
    )
    msi.update_coordinates(aligned_xy[:, 0], aligned_xy[:, 1])

    adata = anndata.AnnData(
        X=feature_values.values.astype(np.float32),
        var=feature_values.columns.to_frame(name="metabolite").set_index("metabolite"),
    )
    adata.obsm["spatial"] = aligned_xy

    table = TableModel.parse(adata)
    image_element = Image2DModel.parse(np.moveaxis(np.atleast_3d(image), -1, 0))

    return SpatialData(images={"he": image_element}, tables={"msi": table})
