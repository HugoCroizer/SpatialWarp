"""Align a set of point coordinates onto their own reference image.

Some spatial technologies (e.g. MSI instrument rasters) report spot
coordinates in a system that does not natively share an origin, scale, or
rotation with the pixel space of their own reference image. ``run_grid_alignment``
is a thin convenience wrapper composing the package's reusable primitives —
:func:`spatialwarp.landmark_picker.pick_landmarks`,
:func:`spatialwarp.registration.register_elastic`, and
:func:`spatialwarp.raster.rasterize_points` — for exactly that case. Nothing
here is technology-specific; the same primitives are used for image-vs-image
cross-modality registration in :mod:`spatialwarp.pipeline`.
"""

import numpy as np
import pandas as pd
from spatialdata import SpatialData

from .raster import rasterize_points, points_to_pixel
from .landmark_picker import pick_landmarks
from .registration import register_elastic
from .spatialdata_builder import extract_image, extract_table, table_to_frame

__all__ = ["rasterize_points", "points_to_pixel", "run_grid_alignment", "align_grid"]


def _intensity_for_feature(points_xy, values, feature):
    if values is None:
        return np.ones(len(points_xy))
    if isinstance(values, pd.DataFrame):
        return values[feature].values if feature is not None else values.sum(axis=1).values
    intensity = np.asarray(values, dtype=float)
    if intensity.ndim > 1:
        intensity = intensity.sum(axis=1)
    return intensity


def run_grid_alignment(
    image,
    points_xy,
    values=None,
    feature=None,
    output_csv=None,
    x_flip=False,
    raster_upsample=1,
    mesh_size=(8, 8),
    number_of_iterations=100,
    pick_landmarks_interactively=True,
    moving_landmarks=None,
    fixed_landmarks=None,
    cmap="viridis",
):
    """Align ``points_xy`` (and their ``values``) onto ``image``'s pixel space.

    Rasterizes the points/values into a pseudo-image on their own native
    raster grid, registers that pseudo-image against ``image`` (landmark-
    guided elastic B-spline registration via SimpleITK), and warps the
    original point coordinates through the resulting transform.

    Parameters
    ----------
    image : np.ndarray
        Reference image (e.g. the section's own H&E scan).
    points_xy : array-like of shape (n, 2)
        Point coordinates in the instrument's own raster space.
    values : pandas.DataFrame, array-like, or None
        Per-point feature value(s) used to build the rasterized pseudo-image.
        If a DataFrame with multiple columns and ``feature`` is not given,
        the interactive picker lets you page through columns to find the
        clearest contrast (see :func:`spatialwarp.landmark_picker.pick_landmarks`);
        non-interactively they're summed into one intensity image. If None,
        all points get equal weight (a binary occupancy image) —
        registration then relies entirely on landmarks.
    feature : str or None
        If ``values`` is a DataFrame, use this column as the rasterized
        intensity instead of summing all columns or browsing interactively —
        useful once you already know which feature gives the best contrast
        (e.g. from a previous interactive run) and want a reproducible,
        non-interactive call.
    output_csv : str or None
        If given, the aligned coordinates are saved here.
    x_flip : bool
        Mirror the x coordinate before aligning (needed by some MSI
        instruments whose raster x-axis runs opposite to the image's).
    raster_upsample : int
        Optional upsampling of the pseudo-image resolution beyond the raw
        raster pitch (can help registration if the raster is very coarse).
    mesh_size, number_of_iterations :
        Passed through to :func:`spatialwarp.registration.register_elastic`.
    pick_landmarks_interactively : bool
        If True (default) and no landmarks are supplied, opens
        :func:`spatialwarp.landmark_picker.pick_landmarks` to seed the affine
        initialization.
    moving_landmarks, fixed_landmarks : array-like of shape (N, 2), or None
        Pre-picked landmarks (moving = ``image``, fixed = the rasterized
        pseudo-image), skipping the interactive picker if given.
    cmap : str
        Colormap for the picker's rasterized pseudo-image, if it's used.

    Returns
    -------
    np.ndarray of shape (n, 2)
        The aligned point coordinates, in ``image``'s pixel space.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    if x_flip:
        points_xy = points_xy.copy()
        points_xy[:, 0] = points_xy[:, 0].max() - points_xy[:, 0]

    needs_picking = moving_landmarks is None and fixed_landmarks is None and pick_landmarks_interactively
    picked_already = False

    if needs_picking and feature is None:
        # Let the user browse features (if there's more than one) while picking
        # landmarks against the rasterized grid directly.
        moving_landmarks, fixed_landmarks, _, feature = pick_landmarks(
            image, (points_xy, values), upsample=raster_upsample, cmap=cmap
        )
        picked_already = True

    intensity = _intensity_for_feature(points_xy, values, feature)
    pseudo_image, transform = rasterize_points(points_xy, intensity, upsample=raster_upsample)

    if needs_picking and not picked_already:
        # feature was already known (explicitly passed in) -- just click landmarks
        # against that fixed pseudo-image, no need for the feature browser
        moving_landmarks, fixed_landmarks, _, _ = pick_landmarks(image, pseudo_image, cmap=cmap)

    result = register_elastic(
        moving_image=image,
        fixed_image=pseudo_image,
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        mesh_size=mesh_size,
        number_of_iterations=number_of_iterations,
    )

    pixel_xy = points_to_pixel(points_xy, transform)
    aligned_x, aligned_y = result.warp_points_fixed_to_moving(pixel_xy[:, 0], pixel_xy[:, 1])
    aligned_xy = np.column_stack([aligned_x, aligned_y])

    if output_csv is not None:
        df_out = pd.DataFrame(
            {
                "index": np.arange(len(aligned_xy)),
                "x_transformed": aligned_xy[:, 0],
                "y_transformed": aligned_xy[:, 1],
            }
        )
        df_out.to_csv(output_csv, index=False)

    return aligned_xy


def align_grid(
    sdata,
    image_key=None,
    table_key=None,
    feature=None,
    output_csv=None,
    raster_upsample=1,
    mesh_size=(8, 8),
    number_of_iterations=100,
    pick_landmarks_interactively=True,
    moving_landmarks=None,
    fixed_landmarks=None,
    cmap="viridis",
):
    """Align a SpatialData object's own points onto its own image.

    The SpatialData-native counterpart to :func:`run_grid_alignment`: give it a
    SpatialData object built directly from raw data (e.g. via
    :func:`spatialwarp.spatialdata_builder.build_spatialdata`) whose
    ``obsm['spatial']`` still holds *unaligned* points (in the technology's own
    raw coordinate system, not yet matching the image's pixel space), and get
    back a new SpatialData object with ``obsm['spatial']`` replaced by the
    aligned coordinates. The original coordinates are kept in
    ``obsm['spatial_raw']`` for reference.

    Any vendor-specific coordinate quirks (e.g. a flipped axis) should be
    resolved before building ``sdata`` — this function only ever deals with
    SpatialData objects, not raw arrays.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Must have one image element and one table element with
        ``obsm['spatial']`` holding the raw, unaligned point coordinates in
        the same units as the image.
    image_key, table_key : str or None
        Element keys to use, only needed if ``sdata`` has more than one
        image/table element.
    feature, output_csv, raster_upsample, mesh_size, number_of_iterations,
    pick_landmarks_interactively, moving_landmarks, fixed_landmarks, cmap :
        Passed through to :func:`run_grid_alignment` — see there for details.

    Returns
    -------
    spatialdata.SpatialData
        A new SpatialData object (same image, same features) with
        ``obsm['spatial']`` replaced by the aligned coordinates.
    """
    image = extract_image(sdata, image_key)
    table = extract_table(sdata, table_key)
    values = table_to_frame(table)
    points_xy = np.asarray(table.obsm["spatial"], dtype=float)

    resolved_image_key = image_key or next(iter(sdata.images))
    resolved_table_key = table_key or next(iter(sdata.tables))

    aligned_xy = run_grid_alignment(
        image=image,
        points_xy=points_xy,
        values=values,
        feature=feature,
        output_csv=output_csv,
        raster_upsample=raster_upsample,
        mesh_size=mesh_size,
        number_of_iterations=number_of_iterations,
        pick_landmarks_interactively=pick_landmarks_interactively,
        moving_landmarks=moving_landmarks,
        fixed_landmarks=fixed_landmarks,
        cmap=cmap,
    )

    new_table = table.copy()
    new_table.obsm["spatial_raw"] = points_xy
    new_table.obsm["spatial"] = aligned_xy

    return SpatialData(
        images={resolved_image_key: sdata.images[resolved_image_key]},
        tables={resolved_table_key: new_table},
    )
