"""Rasterize a point cloud onto its own native pixel grid.

Some spatial technologies (e.g. MSI instrument rasters) report spot
coordinates in a system that does not natively share an origin, scale, or
rotation with the pixel space of a reference image. This module converts
such a point/intensity cloud into a pseudo-image on its own native grid, so
it can be treated identically to a real image everywhere else in the package
(landmark picking, elastic registration, ...).
"""

import numpy as np
from scipy.spatial import cKDTree


def _infer_pitch(points_xy):
    """Estimate the raster's native point spacing as the median nearest-
    neighbor distance.

    Measuring the smallest gap between unique x (or y) coordinates instead
    would only work for a perfectly axis-aligned raster; as soon as the raw
    instrument grid is rotated relative to the x/y axes (common in practice),
    every point gets a near-unique x and y coordinate and that heuristic
    collapses to a near-zero step, producing a pseudo-image with billions of
    pixels. Nearest-neighbor distance is rotation-invariant.
    """
    if len(points_xy) < 2:
        return 1.0
    tree = cKDTree(points_xy)
    dists, _ = tree.query(points_xy, k=2)
    return float(np.median(dists[:, 1]))


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

    step = _infer_pitch(points_xy) / upsample
    step_x = step_y = step

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


def robust_clim(image):
    """1st/99th percentile color limits, robust to outliers and degenerate
    (constant) images."""
    vmin, vmax = np.nanpercentile(image, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1
    return vmin, vmax
