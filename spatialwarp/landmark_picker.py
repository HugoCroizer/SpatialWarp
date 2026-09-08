"""Interactive landmark picker: click corresponding points between two sides.

Each side is either a plain reference image, or a point cloud (rasterized on
the fly onto its own native grid, see :mod:`spatialwarp.raster`) — the same
picker works for image-vs-image (cross-modality registration) and
image-vs-grid (a technology's own points against its own reference image),
regardless of what technology or feature values are involved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .raster import rasterize_points, robust_clim

_SUM_LABEL = "All features (sum)"


class _Side:
    """Displays one side of the picker: a plain image, or a rasterized point
    cloud with an optional feature browser.

    Also tracks a display-only orientation (90-degree rotation + horizontal
    mirror, toggled via the Rotate/Mirror buttons) used purely to make
    landmark selection easier when the two sides start out in very different
    orientations. Clicks are always converted back to this side's original,
    untransformed pixel space before being stored, so the orientation toggle
    never affects the coordinates this module returns.
    """

    def __init__(self, ax, data, upsample, cmap, label):
        self.ax = ax
        self.upsample = upsample
        self.cmap = cmap
        self.is_grid = isinstance(data, tuple)
        self.rotate_k = 0
        self.mirror = False

        if self.is_grid:
            points_xy, values = data
            self.points_xy = np.asarray(points_xy, dtype=float)
            self.values, self.feature_names = self._normalize_values(values)
            self.current_idx = 0
            self._raw_image = self._rasterize(self.current_idx)
        else:
            self._raw_image = np.asarray(data)
            self._label = label

        self.orig_shape = self._raw_image.shape[:2]
        display_image = self._oriented(self._raw_image)

        if self.is_grid:
            vmin, vmax = robust_clim(display_image)
            self.im = ax.imshow(display_image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(self.feature_names[self.current_idx])
        else:
            self.im = ax.imshow(display_image, cmap="gray")
            ax.set_title(label)

    @staticmethod
    def _normalize_values(values):
        if values is None:
            return None, [_SUM_LABEL]
        if isinstance(values, pd.DataFrame):
            if values.shape[1] > 1:
                return values, [_SUM_LABEL] + list(values.columns)
            return values, [values.columns[0]]
        return pd.DataFrame({"value": np.asarray(values, dtype=float)}), ["value"]

    def _intensity(self, idx):
        if self.values is None:
            return np.ones(len(self.points_xy))
        name = self.feature_names[idx]
        if name == _SUM_LABEL:
            return self.values.sum(axis=1).values
        return self.values[name].values

    def _rasterize(self, idx):
        image, self.transform = rasterize_points(self.points_xy, self._intensity(idx), upsample=self.upsample)
        return image

    def _oriented(self, image):
        """Apply this side's current display-only mirror + rotation."""
        if self.mirror:
            image = image[:, ::-1, ...] if image.ndim == 3 else image[:, ::-1]
        if self.rotate_k:
            image = np.rot90(image, k=self.rotate_k)
        return image

    def _refresh_image(self):
        display_image = self._oriented(self._raw_image)
        self.im.set_data(display_image)
        if self.is_grid:
            vmin, vmax = robust_clim(display_image)
            self.im.set_clim(vmin=vmin, vmax=vmax)
        # set_data() doesn't update extent/limits on its own, and rotating by
        # 90/270 swaps height and width -- fix both up so click coordinates
        # keep lining up with the displayed pixels.
        h, w = display_image.shape[:2]
        self.im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)

    def has_feature_browser(self):
        return self.is_grid and len(self.feature_names) > 1

    def cycle_feature(self, direction):
        self.current_idx = (self.current_idx + direction) % len(self.feature_names)
        self._raw_image = self._rasterize(self.current_idx)
        self._refresh_image()
        self.ax.set_title(self.feature_names[self.current_idx])

    def cycle_rotate(self):
        self.rotate_k = (self.rotate_k + 1) % 4
        self._refresh_image()

    def toggle_mirror(self):
        self.mirror = not self.mirror
        self._refresh_image()

    def to_original(self, dispx, dispy):
        """Map a click in the current *display* space back to this side's
        original, untransformed pixel space (what rasterize_points/
        points_to_pixel would produce, or the raw image's own array
        indices) -- i.e. undo whatever rotate/mirror is currently active."""
        h, w = self.orig_shape
        k = self.rotate_k
        if k == 0:
            px, py = dispx, dispy
        elif k == 1:
            px, py = w - 1 - dispy, dispx
        elif k == 2:
            px, py = w - 1 - dispx, h - 1 - dispy
        else:
            px, py = dispy, h - 1 - dispx
        if self.mirror:
            px = (w - 1) - px
        return px, py

    def to_display(self, origx, origy):
        """Inverse of :meth:`to_original`: map a point in this side's
        original pixel space to where it currently appears on screen."""
        h, w = self.orig_shape
        px = (w - 1 - origx) if self.mirror else origx
        py = origy
        k = self.rotate_k
        if k == 0:
            return px, py
        elif k == 1:
            return py, w - 1 - px
        elif k == 2:
            return w - 1 - px, h - 1 - py
        else:
            return h - 1 - py, px

    def chosen_feature(self):
        if not self.is_grid:
            return None
        name = self.feature_names[self.current_idx]
        return None if name == _SUM_LABEL else name


def _add_feature_buttons(fig, side, prev_pos, next_pos, on_change):
    prev_ax = plt.axes(prev_pos)
    next_ax = plt.axes(next_pos)
    prev_btn = Button(prev_ax, "◄ Feature", color="lightblue", hovercolor="skyblue")
    next_btn = Button(next_ax, "Feature ►", color="lightblue", hovercolor="skyblue")

    def _prev(event):
        side.cycle_feature(-1)
        on_change()

    def _next(event):
        side.cycle_feature(1)
        on_change()

    prev_btn.on_clicked(_prev)
    next_btn.on_clicked(_next)
    return prev_btn, next_btn


def _add_orientation_buttons(fig, side, rotate_pos, mirror_pos, on_change):
    rotate_ax = plt.axes(rotate_pos)
    mirror_ax = plt.axes(mirror_pos)
    rotate_btn = Button(rotate_ax, "⟲ Rotate", color="lightyellow", hovercolor="khaki")
    mirror_btn = Button(mirror_ax, "⇄ Mirror", color="lightyellow", hovercolor="khaki")

    def _rotate(event):
        side.cycle_rotate()
        on_change()

    def _mirror(event):
        side.toggle_mirror()
        on_change()

    rotate_btn.on_clicked(_rotate)
    mirror_btn.on_clicked(_mirror)
    return rotate_btn, mirror_btn


def _redraw_markers(ax, side, points_orig, artists):
    for artist in artists:
        artist.remove()
    artists.clear()
    for i, (ox, oy) in enumerate(points_orig, start=1):
        dx, dy = side.to_display(ox, oy)
        (line,) = ax.plot(dx, dy, "r+", markersize=12, mew=2)
        text = ax.annotate(str(i), (dx, dy), color="red")
        artists.append(line)
        artists.append(text)


def pick_landmarks(moving, fixed, upsample=1, cmap="viridis", output_csv=None):
    """Click corresponding landmark points between ``moving`` and ``fixed``.

    Each of ``moving``/``fixed`` is either:

    - a plain image array (2D or 3D), shown as-is, or
    - a ``(points_xy, values)`` pair — a point cloud rasterized on the fly
      onto its own native grid (:func:`spatialwarp.raster.rasterize_points`).
      If ``values`` is a DataFrame with more than one column, "◄ Feature /
      Feature ►" buttons let you page through columns (e.g. individual
      metabolites) to find the clearest contrast; switching the displayed
      feature never invalidates landmarks already picked, since the
      rasterized pixel *positions* only depend on the points' spacing, not
      on which feature is used for pixel brightness.

    Every side also gets "⟲ Rotate" (steps through 0/90/180/270 degrees) and
    "⇄ Mirror" (horizontal flip) buttons, independently. Use these if the two
    sides start out in very different orientations, to make corresponding
    features easier to spot before you start clicking — landmarks already
    placed on a side are redrawn to track its current orientation, and
    whatever is on screen when you close the window has no effect on the
    returned coordinates: clicks are always converted back to that side's
    original, untransformed pixel space immediately.

    This is the same function whether you're aligning two real images
    (cross-modality registration) or a technology's own point grid against
    its own reference image (grid self-alignment) — nothing here is
    technology-specific.

    Parameters
    ----------
    moving, fixed : np.ndarray or (points_xy, values)
    upsample : int
        Passed to :func:`spatialwarp.raster.rasterize_points` for any side
        that's a point cloud.
    cmap : str
        Colormap for any side that's a rasterized point cloud (plain image
        sides always use grayscale).
    output_csv : str or None
        If given, save the picked landmark points to this CSV.

    Returns
    -------
    moving_xy, fixed_xy : np.ndarray of shape (N, 2)
        Picked landmark points, in each side's own original pixel space (for
        a grid side, that's the rasterized pseudo-image's pixel space —
        matching what :func:`spatialwarp.raster.rasterize_points` produces
        if you rasterize the same points again for registration), regardless
        of what rotation/mirror was toggled on screen while picking.
    moving_feature, fixed_feature : str or None
        Which feature was on screen for that side when the window was
        closed, or ``None`` if that side isn't a multi-feature grid (or its
        summed view was shown).
    """
    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.82)

    moving_side = _Side(ax_m, moving, upsample, cmap, "Moving — click a point, then its match on the right")
    fixed_side = _Side(ax_f, fixed, upsample, cmap, "Fixed")

    moving_points, fixed_points = [], []
    moving_artists, fixed_artists = [], []
    pending = {"side": "moving"}

    def _toolbar_busy():
        # the toolbar's own zoom-rect/pan drag starts with a button_press_event too --
        # skip landmark placement while one of those tools is active so dragging to zoom
        # doesn't also drop a spurious landmark at the click location.
        toolbar = fig.canvas.toolbar
        return toolbar is not None and toolbar.mode != ""

    def on_click(event):
        if _toolbar_busy():
            return
        if event.inaxes is ax_m and pending["side"] == "moving" and event.xdata is not None:
            moving_points.append(moving_side.to_original(event.xdata, event.ydata))
            _redraw_markers(ax_m, moving_side, moving_points, moving_artists)
            pending["side"] = "fixed"
            fig.canvas.draw_idle()
        elif event.inaxes is ax_f and pending["side"] == "fixed" and event.xdata is not None:
            fixed_points.append(fixed_side.to_original(event.xdata, event.ydata))
            _redraw_markers(ax_f, fixed_side, fixed_points, fixed_artists)
            pending["side"] = "moving"
            fig.canvas.draw_idle()

    def on_scroll(event):
        # mouse-wheel zoom centered on the cursor -- independent of the toolbar's
        # click-drag zoom tool, and never interferes with landmark placement since it's
        # a different event type entirely.
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x, y = event.xdata, event.ydata
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
        ax.set_ylim(y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    def _moving_changed():
        _redraw_markers(ax_m, moving_side, moving_points, moving_artists)
        fig.canvas.draw_idle()

    def _fixed_changed():
        _redraw_markers(ax_f, fixed_side, fixed_points, fixed_artists)
        fig.canvas.draw_idle()

    buttons = []  # keep references alive (matplotlib widgets need a live reference)
    if moving_side.has_feature_browser():
        buttons += _add_feature_buttons(
            fig, moving_side, [0.12, 0.93, 0.09, 0.04], [0.22, 0.93, 0.09, 0.04], _moving_changed
        )
    if fixed_side.has_feature_browser():
        buttons += _add_feature_buttons(
            fig, fixed_side, [0.55, 0.93, 0.09, 0.04], [0.65, 0.93, 0.09, 0.04], _fixed_changed
        )
    buttons += _add_orientation_buttons(
        fig, moving_side, [0.12, 0.865, 0.09, 0.04], [0.22, 0.865, 0.09, 0.04], _moving_changed
    )
    buttons += _add_orientation_buttons(
        fig, fixed_side, [0.55, 0.865, 0.09, 0.04], [0.65, 0.865, 0.09, 0.04], _fixed_changed
    )

    plt.show(block=True)

    n = min(len(moving_points), len(fixed_points))
    moving_xy = np.array(moving_points[:n], dtype=float).reshape(-1, 2)
    fixed_xy = np.array(fixed_points[:n], dtype=float).reshape(-1, 2)

    if output_csv is not None:
        pd.DataFrame(
            {
                "moving_x": moving_xy[:, 0],
                "moving_y": moving_xy[:, 1],
                "fixed_x": fixed_xy[:, 0],
                "fixed_y": fixed_xy[:, 1],
            }
        ).to_csv(output_csv, index=False)

    return moving_xy, fixed_xy, moving_side.chosen_feature(), fixed_side.chosen_feature()
