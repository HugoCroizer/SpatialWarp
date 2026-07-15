"""Align a set of point coordinates onto their own reference image.

Some spatial technologies (e.g. MSI instrument rasters) report spot
coordinates in a system that does not natively share an origin, scale, or
rotation with the pixel space of their own reference image. This module
handles that by rasterizing the point/intensity data onto its own native
pixel grid (inferred from the points' spacing) to get a pseudo-image, then
running the same landmark-guided elastic registration used for cross-modality
alignment (:mod:`spatialwarp.registration`) between that pseudo-image and the
real reference image. This reduces "align a grid to its own image" to the
same machinery as "align two different images" (see
:mod:`spatialwarp.registration` for that).
"""

import numpy as np
import pandas as pd

from .landmark_picker import pick_landmarks
from .registration import register_elastic


def rasterize_points(points_xy, values, upsample=1):
    """Bin ``(points_xy, values)`` onto a regular pixel grid inferred from
    the points' own native spacing.

    Parameters
    ----------
    points_xy : array-like of shape (n, 2)
    values : array-like of shape (n,)
    upsample : int
        Optional upsampling of the pixel grid beyond the raw raster pitch.

    Returns
    -------
    image : np.ndarray of shape (H, W)
    transform : tuple (min_x, step_x, min_y, step_y)
        Converts original point coordinates to pixel indices in ``image``
        via :func:`points_to_pixel`.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    values = np.asarray(values, dtype=float)
    x, y = points_xy[:, 0], points_xy[:, 1]

    ux = np.unique(np.round(x, 6))
    uy = np.unique(np.round(y, 6))
    step_x = np.min(np.diff(np.sort(ux))) if len(ux) > 1 else 1.0
    step_y = np.min(np.diff(np.sort(uy))) if len(uy) > 1 else 1.0
    step_x /= upsample
    step_y /= upsample

    min_x, min_y = x.min(), y.min()
    px = np.round((x - min_x) / step_x).astype(int)
    py = np.round((y - min_y) / step_y).astype(int)

    width, height = px.max() + 1, py.max() + 1
    image = np.zeros((height, width), dtype=float)
    image[py, px] = values

    return image, (min_x, step_x, min_y, step_y)


def points_to_pixel(points_xy, transform):
    """Convert original point coordinates to pixel indices using a
    ``transform`` returned by :func:`rasterize_points`."""
    min_x, step_x, min_y, step_y = transform
    points_xy = np.asarray(points_xy, dtype=float)
    px = (points_xy[:, 0] - min_x) / step_x
    py = (points_xy[:, 1] - min_y) / step_y
    return np.column_stack([px, py])


def run_grid_alignment(
    image,
    points_xy,
    values=None,
    output_csv=None,
    x_flip=False,
    raster_upsample=1,
    mesh_size=(8, 8),
    number_of_iterations=100,
    pick_landmarks_interactively=True,
    moving_landmarks=None,
    fixed_landmarks=None,
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
        If a DataFrame with multiple columns, they're summed into one
        intensity image. If None, all points get equal weight (a binary
        occupancy image) — registration then relies entirely on landmarks.
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
        If True (default) and no landmarks are supplied, opens an interactive
        picker (:func:`spatialwarp.landmark_picker.pick_landmarks`) to seed
        the affine initialization.
    moving_landmarks, fixed_landmarks : array-like of shape (N, 2), or None
        Pre-picked landmarks (moving = ``image``, fixed = the rasterized
        pseudo-image), skipping the interactive picker if given.

    Returns
    -------
    np.ndarray of shape (n, 2)
        The aligned point coordinates, in ``image``'s pixel space.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    if x_flip:
        points_xy = points_xy.copy()
        points_xy[:, 0] = points_xy[:, 0].max() - points_xy[:, 0]

    if values is None:
        intensity = np.ones(len(points_xy))
    elif isinstance(values, pd.DataFrame):
        intensity = values.sum(axis=1).values
    else:
        intensity = np.asarray(values, dtype=float)
        if intensity.ndim > 1:
            intensity = intensity.sum(axis=1)

    pseudo_image, transform = rasterize_points(points_xy, intensity, upsample=raster_upsample)

    if moving_landmarks is None and fixed_landmarks is None and pick_landmarks_interactively:
        moving_landmarks, fixed_landmarks = pick_landmarks(image, pseudo_image)

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
