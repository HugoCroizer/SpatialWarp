"""QC visualization: overlay points on a reference image with an alpha slider.

Useful after either alignment step — :func:`spatialwarp.grid_align.run_grid_alignment`
(check the MSI grid lines up with its own H&E image) or
:func:`spatialwarp.pipeline.align` (check ``adata.obsm['spatial_warped']``
lines up with the moving modality's own points) — to visually inspect overlap
quality by eye.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_overlay(image, points_xy, values=None, point_size=3, cmap="hot", initial_alpha=0.5):
    """Show ``points_xy`` scattered on top of ``image``, with a slider to
    adjust the points' alpha so you can inspect how well they line up.

    Parameters
    ----------
    image : np.ndarray
        Reference image to overlay points on.
    points_xy : array-like of shape (n, 2)
        Point coordinates in the same pixel space as ``image``.
    values : array-like of shape (n,), or None
        Optional per-point values for color-coding (e.g. total intensity).
        If None, points are plotted in a single color.
    point_size : float
    cmap : str
    initial_alpha : float

    Returns
    -------
    (fig, ax) : the created matplotlib Figure and Axes.
    """
    points_xy = np.asarray(points_xy, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    scatter_kwargs = dict(s=point_size, alpha=initial_alpha)
    if values is not None:
        scatter_kwargs.update(c=np.asarray(values, dtype=float), cmap=cmap)
    else:
        scatter_kwargs.update(c="red")

    pts = ax.scatter(points_xy[:, 0], points_xy[:, 1], **scatter_kwargs)
    ax.set_title("Alignment QC — drag the slider to inspect overlap")
    ax.set_aspect("equal")

    plt.subplots_adjust(bottom=0.15)
    slider_ax = plt.axes([0.2, 0.03, 0.6, 0.03])
    s_alpha = Slider(slider_ax, "Alpha", 0.0, 1.0, valinit=initial_alpha)

    def update(val):
        pts.set_alpha(s_alpha.val)
        fig.canvas.draw_idle()

    s_alpha.on_changed(update)
    plt.show(block=True)

    return fig, ax
