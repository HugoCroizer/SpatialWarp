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
import matplotlib.pyplot as plt

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


def pick_landmarks_multi_feature(image, points_xy, values, upsample=1, initial_feature=None, cmap="viridis"):
    """Like :func:`spatialwarp.landmark_picker.pick_landmarks`, but for picking
    landmarks against a rasterized point cloud with more than one feature
    column (e.g. several metabolites) — Prev/Next buttons let you page
    through features to find the one with the clearest contrast, before or
    while clicking landmarks.

    Switching the displayed feature does not invalidate landmarks already
    picked: the rasterized pixel *positions* only depend on the points'
    spacing, not on which feature is used for pixel brightness, so a click
    at a given pixel means the same thing regardless of which feature is
    currently shown.

    Parameters
    ----------
    image : np.ndarray
        Reference image (moving side), shown on the left, static.
    points_xy : array-like of shape (n, 2)
    values : pandas.DataFrame
        Per-point feature columns to page through for the rasterized
        pseudo-image (right side).
    upsample : int
        Passed through to :func:`rasterize_points`.
    initial_feature : str or None
        Column to show first. Defaults to the first column.
    cmap : str
        Colormap for the rasterized pseudo-image (right side).

    Returns
    -------
    (moving_xy, fixed_xy, feature) : np.ndarray, np.ndarray, str or None
        Picked landmarks (same shapes as ``pick_landmarks``) plus the name of
        the feature that was on screen when the window was closed, or
        ``None`` if it was the "All features (sum)" option (matching
        :func:`run_grid_alignment`'s ``feature=None`` meaning "sum them all").
    """
    SUM_LABEL = "All features (sum)"
    feature_names = [SUM_LABEL] + list(values.columns)
    current = {"idx": 0 if initial_feature is None else feature_names.index(initial_feature)}

    def intensity_for(idx):
        if feature_names[idx] == SUM_LABEL:
            return values.sum(axis=1).values
        return values[feature_names[idx]].values

    def rasterized(idx):
        pseudo, _ = rasterize_points(points_xy, intensity_for(idx), upsample=upsample)
        return pseudo

    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.88)
    ax_m.imshow(image, cmap="gray")
    ax_m.set_title("Moving — click a point, then its match on the right")

    pseudo0 = rasterized(current["idx"])
    vmin0, vmax0 = np.nanpercentile(pseudo0, [1, 99])
    if vmax0 <= vmin0:
        vmax0 = vmin0 + 1
    im_f = ax_f.imshow(pseudo0, cmap=cmap, vmin=vmin0, vmax=vmax0)
    ax_f.set_title(feature_names[current["idx"]])

    moving_points, fixed_points = [], []
    pending = {"side": "moving"}

    def on_click(event):
        if event.inaxes is ax_m and pending["side"] == "moving" and event.xdata is not None:
            moving_points.append((event.xdata, event.ydata))
            ax_m.plot(event.xdata, event.ydata, "r+", markersize=12, mew=2)
            ax_m.annotate(str(len(moving_points)), (event.xdata, event.ydata), color="red")
            pending["side"] = "fixed"
            fig.canvas.draw_idle()
        elif event.inaxes is ax_f and pending["side"] == "fixed" and event.xdata is not None:
            fixed_points.append((event.xdata, event.ydata))
            ax_f.plot(event.xdata, event.ydata, "r+", markersize=12, mew=2)
            ax_f.annotate(str(len(fixed_points)), (event.xdata, event.ydata), color="red")
            pending["side"] = "moving"
            fig.canvas.draw_idle()

    def update_feature(direction):
        current["idx"] = (current["idx"] + direction) % len(feature_names)
        data = rasterized(current["idx"])
        im_f.set_data(data)
        vmin, vmax = np.nanpercentile(data, [1, 99])
        if vmax <= vmin:
            vmax = vmin + 1
        im_f.set_clim(vmin=vmin, vmax=vmax)
        ax_f.set_title(feature_names[current["idx"]])
        fig.canvas.draw_idle()

    from matplotlib.widgets import Button

    button_prev_ax = plt.axes([0.55, 0.93, 0.1, 0.04])
    button_next_ax = plt.axes([0.66, 0.93, 0.1, 0.04])
    button_prev = Button(button_prev_ax, "◄ Feature", color="lightblue", hovercolor="skyblue")
    button_next = Button(button_next_ax, "Feature ►", color="lightblue", hovercolor="skyblue")
    button_prev.on_clicked(lambda event: update_feature(-1))
    button_next.on_clicked(lambda event: update_feature(1))

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    n = min(len(moving_points), len(fixed_points))
    moving_xy = np.array(moving_points[:n], dtype=float).reshape(-1, 2)
    fixed_xy = np.array(fixed_points[:n], dtype=float).reshape(-1, 2)

    chosen = feature_names[current["idx"]]
    return moving_xy, fixed_xy, (None if chosen == SUM_LABEL else chosen)


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
        an interactive Prev/Next feature browser lets you pick which column
        gives the clearest contrast before/while placing landmarks (see
        :func:`pick_landmarks_multi_feature`); non-interactively they're
        summed into one intensity image. If None, all points get equal
        weight (a binary occupancy image) — registration then relies
        entirely on landmarks.
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
        If True (default) and no landmarks are supplied, opens an interactive
        picker to seed the affine initialization — the feature-browsing one
        if ``values`` is a multi-column DataFrame and ``feature`` isn't set,
        otherwise :func:`spatialwarp.landmark_picker.pick_landmarks`.
    moving_landmarks, fixed_landmarks : array-like of shape (N, 2), or None
        Pre-picked landmarks (moving = ``image``, fixed = the rasterized
        pseudo-image), skipping the interactive picker if given.
    cmap : str
        Colormap for the feature-browsing picker's pseudo-image, if it's used.

    Returns
    -------
    np.ndarray of shape (n, 2)
        The aligned point coordinates, in ``image``'s pixel space.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    if x_flip:
        points_xy = points_xy.copy()
        points_xy[:, 0] = points_xy[:, 0].max() - points_xy[:, 0]

    is_multi_feature = isinstance(values, pd.DataFrame) and values.shape[1] > 1

    if (
        is_multi_feature
        and feature is None
        and moving_landmarks is None
        and fixed_landmarks is None
        and pick_landmarks_interactively
    ):
        moving_landmarks, fixed_landmarks, feature = pick_landmarks_multi_feature(
            image, points_xy, values, upsample=raster_upsample, cmap=cmap
        )

    if values is None:
        intensity = np.ones(len(points_xy))
    elif isinstance(values, pd.DataFrame):
        intensity = values[feature].values if feature is not None else values.sum(axis=1).values
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
