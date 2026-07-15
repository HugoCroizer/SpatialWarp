"""Read BUnwarpJ transform files and apply them to images and points."""

from itertools import islice

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def read_bunwarp_matrix(path):
    """Parse a BUnwarpJ ``*_direct_transf`` / ``*_inverse_transf`` file.

    The file has a two-line header (``Width=``, ``Height=``) followed by two
    whitespace-separated matrices (x-displacement then y-displacement), each
    ``height`` rows by ``width`` columns.

    Returns
    -------
    (matrix_x, matrix_y) : np.ndarray, np.ndarray
        Each of shape ``(width, height)`` (transposed from the file's
        row-major ``height x width`` layout, matching the orientation the
        rest of this module expects).
    """
    with open(path) as file:
        header = list(islice(file, 2))

    width = int(header[0].split("=")[1])
    height = int(header[1].split("=")[1])
    matrix_size = np.array([width, height])

    matrix_x = pd.read_csv(
        path, skiprows=4, header=None, nrows=matrix_size[1], sep=r"\s+"
    ).values.T

    matrix_y = pd.read_csv(
        path, skiprows=6 + matrix_size[1], header=None, sep=r"\s+"
    ).values.T

    return matrix_x, matrix_y


def warp_image(matrix_x, matrix_y, source_image, order=1, mode="nearest"):
    """Warp ``source_image`` using a BUnwarpJ direct-transform matrix pair.

    ``matrix_x``/``matrix_y`` (as returned by :func:`read_bunwarp_matrix`,
    already transposed to ``(height, width)``) give, for each pixel of the
    *target* space, the corresponding coordinate in ``source_image``.
    """
    if source_image.ndim == 2:
        source_image = source_image[..., np.newaxis]

    channels = source_image.shape[2]
    H, W = matrix_x.shape

    coords = np.vstack([matrix_y.ravel(), matrix_x.ravel()])

    warped = np.zeros((H, W, channels), dtype=source_image.dtype)
    for c in range(channels):
        warped_ch = map_coordinates(
            source_image[:, :, c], coords, order=order, mode=mode
        )
        warped[:, :, c] = warped_ch.reshape(H, W)

    return warped


def warp_points(matrix_x, matrix_y, x, y, source_shape, order=1, mode="nearest"):
    """Warp point coordinates ``(x, y)`` defined in ``source_shape`` pixel
    space into the BUnwarpJ transform's target space.

    ``matrix_x``/``matrix_y`` must come from the matching *inverse*
    transform file (i.e. the one whose matrix resolution corresponds to the
    target space these points should land in). A scale correction is applied
    for the case where the matrix resolution differs from ``source_shape``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    scale_y = matrix_x.shape[0] / source_shape[0]
    scale_x = matrix_x.shape[1] / source_shape[1]

    x_matrix = x * scale_x
    y_matrix = y * scale_y

    coords = np.vstack([y_matrix, x_matrix])

    x_warped = map_coordinates(matrix_x, coords, order=order, mode=mode)
    y_warped = map_coordinates(matrix_y, coords, order=order, mode=mode)

    return x_warped, y_warped
