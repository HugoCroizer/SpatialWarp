"""Nearest-neighbor matching between two point sets in the same coordinate space."""

import numpy as np
from scipy.spatial import cKDTree


def match_nearest(query_xy, reference_xy, distance_threshold=None):
    """Match each point in ``query_xy`` to its nearest point in ``reference_xy``.

    Parameters
    ----------
    query_xy, reference_xy : array-like of shape (n, 2)
    distance_threshold : float or None
        If given, ``keep_mask`` marks matches whose distance is below this
        threshold.

    Returns
    -------
    indices : np.ndarray
        For each query point, the row index into ``reference_xy`` of its
        nearest match.
    distances : np.ndarray
        Distance to that match.
    keep_mask : np.ndarray of bool
        All ``True`` if ``distance_threshold`` is None, else
        ``distances < distance_threshold``.
    """
    tree = cKDTree(np.asarray(reference_xy))
    distances, indices = tree.query(np.asarray(query_xy), k=1)

    if distance_threshold is None:
        keep_mask = np.ones(len(distances), dtype=bool)
    else:
        keep_mask = distances < distance_threshold

    return indices, distances, keep_mask
