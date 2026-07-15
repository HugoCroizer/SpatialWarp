"""Interactive landmark picker: click corresponding points between two images.

Pure-Python replacement for placing landmarks in Fiji's Point tool — used to
seed :func:`spatialwarp.registration.register_elastic`'s affine initialization.
"""

import numpy as np
import matplotlib.pyplot as plt


def pick_landmarks(moving_image, fixed_image, output_csv=None):
    """Show ``moving_image`` and ``fixed_image`` side by side and collect
    corresponding landmark points by clicking.

    Click a point on the moving (left) image, then click its corresponding
    point on the fixed (right) image; repeat. Close the window when done.

    Parameters
    ----------
    moving_image, fixed_image : np.ndarray
    output_csv : str or None
        If given, save the picked points to this CSV
        (columns: moving_x, moving_y, fixed_x, fixed_y).

    Returns
    -------
    (moving_xy, fixed_xy) : np.ndarray, np.ndarray
        Each of shape (N, 2), same order (point i on one side corresponds to
        point i on the other).
    """
    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(14, 7))
    ax_m.imshow(moving_image, cmap="gray")
    ax_m.set_title("Moving — click a point, then its match on the right")
    ax_f.imshow(fixed_image, cmap="gray")
    ax_f.set_title("Fixed")

    moving_points = []
    fixed_points = []
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

    fig.canvas.mpl_connect("button_press_event", on_click)
    # block=True is required explicitly: under Jupyter's %matplotlib tk/qt,
    # IPython turns on interactive mode (plt.ion()), which makes a plain
    # show() return immediately instead of waiting for the window to close.
    plt.show(block=True)

    n = min(len(moving_points), len(fixed_points))
    moving_xy = np.array(moving_points[:n], dtype=float).reshape(-1, 2)
    fixed_xy = np.array(fixed_points[:n], dtype=float).reshape(-1, 2)

    if n == 0:
        print(
            "Warning: no landmark pairs were picked (window closed before any clicks "
            "registered, or clicks landed outside the axes). Returning empty arrays — "
            "register_elastic() will fall back to image-content-only registration."
        )

    if output_csv is not None:
        import pandas as pd

        pd.DataFrame(
            {
                "moving_x": moving_xy[:, 0],
                "moving_y": moving_xy[:, 1],
                "fixed_x": fixed_xy[:, 0],
                "fixed_y": fixed_xy[:, 1],
            }
        ).to_csv(output_csv, index=False)

    return moving_xy, fixed_xy
