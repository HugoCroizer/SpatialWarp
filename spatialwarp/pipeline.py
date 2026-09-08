"""Generic cross-modality alignment of two SpatialData datasets.

Given a ``moving`` and a ``fixed`` ``spatialdata.SpatialData`` object (each
with one image element and one table element whose ``obsm['spatial']`` holds
pixel coordinates matching that image), this registers the two images via
landmark-guided elastic registration (:mod:`spatialwarp.registration`), warps
``fixed``'s points into ``moving``'s pixel space, nearest-neighbor matches
them (:mod:`spatialwarp.matching`), and returns ``fixed``'s table with
``moving``'s matched features merged in.

This has no notion of "MSI" or "Visium" — either side can be any technology,
as long as its data has been formatted into a SpatialData object first (see
``examples/msi`` for one such example).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import matching, registration
from .spatialdata_builder import extract_image, extract_table


def align(
    moving,
    fixed,
    registration_result=None,
    already_aligned=False,
    moving_landmarks=None,
    fixed_landmarks=None,
    distance_threshold=20.0,
    moving_image_key=None,
    fixed_image_key=None,
    moving_table_key=None,
    fixed_table_key=None,
    mesh_size=(8, 8),
    number_of_iterations=100,
    moving_obsm_key="moving_features",
):
    """Register ``moving``/``fixed`` and merge ``moving``'s matched features
    into ``fixed``'s table.

    Parameters
    ----------
    moving, fixed : spatialdata.SpatialData
    registration_result : spatialwarp.registration.RegistrationResult or None
        If given, skips running registration entirely and uses this
        already-computed result (e.g. to reuse one slide's registration
        across several MSI modalities). If ``None`` (default), registration
        is run via :func:`spatialwarp.registration.register_elastic`. Ignored
        if ``already_aligned`` is True.
    already_aligned : bool
        Set True when ``moving``'s ``obsm['spatial']`` is already expressed
        in ``fixed``'s pixel space — e.g. ``moving`` was built via
        :func:`spatialwarp.spatialdata_builder.build_spatialdata` using
        ``fixed``'s own image as the reference and then run through
        :func:`spatialwarp.grid_align.align_grid`, so its points were
        aligned directly onto ``fixed``'s image rather than onto ``moving``'s
        own separate image. Skips registration entirely (no image is
        extracted from either side) and matches points directly, no warp.
    moving_landmarks, fixed_landmarks : array-like of shape (N, 2), or None
        Corresponding (x, y) landmark points used to seed the affine
        initialization, e.g. from :func:`spatialwarp.landmark_picker.pick_landmarks`.
        Ignored if ``registration_result`` is given or ``already_aligned`` is True.
    distance_threshold : float or None
        Matches farther apart than this (in fixed-image pixels, post-warp)
        are dropped. ``None`` keeps everything.
    moving_image_key, fixed_image_key, moving_table_key, fixed_table_key : str or None
        Element keys to use, only needed if a SpatialData has more than one
        image/table element.
    mesh_size, number_of_iterations :
        Passed through to :func:`spatialwarp.registration.register_elastic`.
    moving_obsm_key : str
        Key under which ``moving``'s matched feature matrix is stored in
        ``merged.obsm`` (as a DataFrame with ``moving``'s var names as
        columns) — e.g. pass ``"metabolite"`` or ``"lipid"`` to keep multiple
        modalities distinguishable when merging more than one into the same
        downstream analysis.

    Returns
    -------
    anndata.AnnData
        ``fixed``'s table, filtered to matched points within
        ``distance_threshold``, with ``moving``'s matched feature matrix in
        ``obsm[moving_obsm_key]``, plus ``nearest_index``/``nearest_distance``.
    """
    moving_table = extract_table(moving, moving_table_key)
    fixed_table = extract_table(fixed, fixed_table_key)

    moving_xy = moving_table.obsm["spatial"]
    fixed_xy = fixed_table.obsm["spatial"]

    if already_aligned:
        x_warped, y_warped = fixed_xy[:, 0], fixed_xy[:, 1]
    else:
        if registration_result is None:
            # Only extract (and, for dask-backed multiscale images, compute) the
            # image arrays when they're actually needed for registration — skip
            # this expensive step entirely when reusing a precomputed result.
            moving_image = extract_image(moving, moving_image_key)
            fixed_image = extract_image(fixed, fixed_image_key)
            registration_result = registration.register_elastic(
                moving_image,
                fixed_image,
                moving_landmarks=moving_landmarks,
                fixed_landmarks=fixed_landmarks,
                mesh_size=mesh_size,
                number_of_iterations=number_of_iterations,
            )

        x_warped, y_warped = registration_result.warp_points_fixed_to_moving(
            fixed_xy[:, 0], fixed_xy[:, 1]
        )

    indices, distances, keep = matching.match_nearest(
        np.column_stack([x_warped, y_warped]), moving_xy, distance_threshold
    )

    merged = fixed_table[keep].copy()
    matched_indices = indices[keep]

    merged.obs["nearest_index"] = matched_indices
    merged.obs["nearest_distance"] = distances[keep]
    merged.obsm["spatial_warped"] = np.column_stack([x_warped, y_warped])[keep]

    matched_X = moving_table.X[matched_indices]
    # moving_table.X is sparse for tables loaded straight from vendor pipelines (e.g. Space
    # Ranger's filtered_feature_bc_matrix.h5), dense for tables built via build_spatialdata().
    # np.asarray() alone silently collapses a sparse matrix into a 0-d object array instead of
    # densifying it, so sparse needs an explicit .toarray() first.
    if sp.issparse(matched_X):
        matched_X = matched_X.toarray()
    else:
        matched_X = np.asarray(matched_X)

    moving_features = pd.DataFrame(
        matched_X,
        columns=moving_table.var_names,
        index=merged.obs.index,
    )
    merged.obsm[moving_obsm_key] = moving_features

    merged.uns["spatialwarp"] = {
        "distance_threshold": distance_threshold,
    }

    return merged
