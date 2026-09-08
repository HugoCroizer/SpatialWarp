"""Fine-tune an existing registration using matched-feature correlation.

:mod:`spatialwarp.registration` and :mod:`spatialwarp.landmark_picker` align two *images*
(landmarks and/or image intensity). Once :func:`spatialwarp.pipeline.align` has transferred
features across, you often already know a marker that *should* correspond between the two
modalities — a gene and the protein it encodes, say — and can check directly whether it lines
up as tightly as it should. This module nudges the registration (a small rigid correction:
rotation + translation, applied on top of the existing affine + bspline) to maximize that
correlation directly, rather than re-picking landmarks.

This is a local refinement, not a replacement for registration — bound the search with
``max_translation``/``max_rotation_deg`` to keep it that way.
"""

import numpy as np
import SimpleITK as sitk
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .raster import _infer_pitch
from .registration import RegistrationResult, register_elastic
from .spatialdata_builder import extract_table


def _feature_values(table, feature_name):
    return np.asarray(table.obs_vector(feature_name), dtype=float)


def _apply_rigid(xy, params, center):
    theta, tx, ty = params
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return (xy - center) @ rotation.T + center + np.array([tx, ty])


def _mean_correlation(indices, keep, moving_values, fixed_values):
    """Mean Pearson correlation across ``feature_pairs``' matched values, or None if there
    aren't enough kept matches (or every pair is degenerate) to trust one."""
    if np.count_nonzero(keep) < 10:
        return None
    correlations = []
    for m_vals, f_vals in zip(moving_values, fixed_values):
        a = m_vals[indices[keep]]
        b = f_vals[keep]
        if a.std() == 0 or b.std() == 0:
            continue
        correlations.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(correlations)) if correlations else None


def _match_and_correlate(moving_tree, warped_xy, distance_threshold, moving_values, fixed_values):
    distances, indices = moving_tree.query(warped_xy, k=1)
    keep = distances < distance_threshold if distance_threshold is not None else np.ones(len(distances), dtype=bool)
    return _mean_correlation(indices, keep, moving_values, fixed_values)


def _rasterize_shared(points_a, values_a, points_b, values_b, upsample=1):
    """Bin two point clouds onto the *same* pixel grid (shared origin, step, size) -- unlike
    :func:`spatialwarp.raster.rasterize_points`, which infers its own grid per point cloud and
    so can't be used to compare two different point sets directly.

    The step is the *coarser* of the two clouds' own native pitch (not the finer one): the
    denser side (typically the ``moving`` feature grid) then has several points per output
    pixel and gets properly averaged rather than a handful of scattered points on an otherwise
    near-empty fine grid, which registration-by-mutual-information handles far better than a
    mostly-zero sparse raster would.

    Returns
    -------
    image_a, image_b : np.ndarray, same shape
    transform : tuple (min_x, step, min_y, step)
        Same convention as :func:`spatialwarp.raster.rasterize_points`'s transform.
    """
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    step = max(_infer_pitch(points_a), _infer_pitch(points_b)) / upsample

    all_xy = np.vstack([points_a, points_b])
    min_x, min_y = all_xy.min(axis=0)
    max_x, max_y = all_xy.max(axis=0)
    width = int(np.round((max_x - min_x) / step)) + 1
    height = int(np.round((max_y - min_y) / step)) + 1

    def _bin(points, values):
        px = np.round((points[:, 0] - min_x) / step).astype(int)
        py = np.round((points[:, 1] - min_y) / step).astype(int)
        sums = np.zeros((height, width), dtype=float)
        counts = np.zeros((height, width), dtype=float)
        np.add.at(sums, (py, px), values)
        np.add.at(counts, (py, px), 1.0)
        with np.errstate(invalid="ignore"):
            image = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return image

    image_a = _bin(points_a, values_a)
    image_b = _bin(points_b, values_b)
    return image_a, image_b, (min_x, step, min_y, step)


def _pixel_physical_transforms(raster_transform):
    """Build the affine pair that converts between a raster's own pixel-index space (what a
    fitted SimpleITK transform over that raster operates in) and the original physical point
    coordinates it was binned from."""
    min_x, step_x, min_y, step_y = raster_transform

    to_pixel = sitk.AffineTransform(2)
    to_pixel.SetMatrix([1.0 / step_x, 0.0, 0.0, 1.0 / step_y])
    to_pixel.SetTranslation((-min_x / step_x, -min_y / step_y))

    to_physical = sitk.AffineTransform(2)
    to_physical.SetMatrix([step_x, 0.0, 0.0, step_y])
    to_physical.SetTranslation((min_x, min_y))

    return to_pixel, to_physical


def refine_alignment(
    moving,
    fixed,
    registration_result,
    feature_pairs,
    moving_table_key=None,
    fixed_table_key=None,
    moving_obsm_key="moving_features",
    distance_threshold=20.0,
    max_translation=50.0,
    max_rotation_deg=5.0,
    min_improvement=0.02,
):
    """Nudge ``registration_result`` to maximize correlation between matched features.

    Parameters
    ----------
    moving, fixed : spatialdata.SpatialData
        Same objects you'd pass to :func:`spatialwarp.pipeline.align`.
    registration_result : spatialwarp.registration.RegistrationResult
        An already-fit registration (from landmarks and/or elastic registration) to refine.
    feature_pairs : list of (str, str)
        ``(moving_feature_name, fixed_feature_name)`` pairs — e.g.
        ``[("aSMA_780 (Opal 780)", "ACTA2")]`` to align a protein channel to the gene that
        encodes it. Each name must be a var name (or obs column) on the respective side's
        table. With more than one pair, the *mean* of their correlations is maximized.
    distance_threshold : float
        Same meaning as in :func:`spatialwarp.pipeline.align` — matches farther apart than
        this (in fixed-image pixels, post-warp) are dropped before computing the
        correlation, so the search isn't rewarded for merely admitting more (weaker) matches.
    max_translation, max_rotation_deg : float
        Bounds on the search, in fixed-image pixels / degrees — keeps this a small local
        nudge rather than a full re-registration. A warning prints if the fit lands at or
        near a bound; raise it and re-run if so.
    min_improvement : float
        The fitted correction's correlation must beat the no-correction baseline by at least
        this much (in raw Pearson-r units) or it's discarded and the original registration is
        returned unchanged. A 3-parameter search over a noisy correlation signal will often
        find a *technically* higher value than the baseline that's really just noise -- e.g. a
        baseline of 0.29 vs a "fitted" 0.30 isn't a correction worth trusting. Set lower only
        if you've separately confirmed small differences in your specific ``feature_pairs``
        are meaningful, not noise.

    Returns
    -------
    merged : anndata.AnnData
        Same shape as :func:`spatialwarp.pipeline.align`'s return, using the refined transform.
    refined_registration_result : spatialwarp.registration.RegistrationResult
        ``registration_result`` with the fitted correction attached — ``.save()`` it like any
        other registration (writes an extra ``.correction.tfm`` alongside the usual two files)
        to reuse the refinement later, e.g. when transferring onto a finer resolution.
    """
    # local import -- avoids a circular import (pipeline imports nothing from here, but this
    # module's only use of pipeline.align is for the final merge, after the search is done)
    from . import pipeline

    moving_table = extract_table(moving, moving_table_key)
    fixed_table = extract_table(fixed, fixed_table_key)

    moving_xy = moving_table.obsm["spatial"]
    fixed_xy = fixed_table.obsm["spatial"]

    moving_values = [_feature_values(moving_table, m) for m, _ in feature_pairs]
    fixed_values = [_feature_values(fixed_table, f) for _, f in feature_pairs]

    x0, y0 = registration_result.warp_points_fixed_to_moving(fixed_xy[:, 0], fixed_xy[:, 1])
    warped0 = np.column_stack([x0, y0])
    center = warped0.mean(axis=0)

    # built once and reused for every candidate below, instead of going through
    # matching.match_nearest (which rebuilds this tree from scratch on every call) -- moving_xy
    # never changes during the search, only the candidate correction applied to warped0 does
    moving_tree = cKDTree(moving_xy)
    max_rotation = np.radians(max_rotation_deg)

    def objective(params):
        theta, tx, ty = params
        penalty = 1e3 * max(0.0, abs(theta) - max_rotation)
        penalty += 1e3 * max(0.0, abs(tx) - max_translation)
        penalty += 1e3 * max(0.0, abs(ty) - max_translation)

        warped = _apply_rigid(warped0, params, center)
        correlation = _match_and_correlate(moving_tree, warped, distance_threshold, moving_values, fixed_values)
        if correlation is None:
            return 1e6  # too few matches to trust a correlation from

        return -correlation + penalty

    # Nearest-neighbor matching makes this objective a step function: a sub-pixel perturbation
    # almost never flips any point's match, so it looks perfectly flat right around x0=[0,0,0]
    # -- Nelder-Mead's default initial simplex (a ~5% relative step, i.e. ~0 here since x0 is
    # exactly zero) would see no signal at all and stall on the first iteration. An explicit
    # initial simplex spanning a real fraction of the search bounds gives it something to
    # actually compare.
    x0 = np.array([0.0, 0.0, 0.0])
    initial_simplex = np.array(
        [
            x0,
            x0 + [max_rotation * 0.3, 0.0, 0.0],
            x0 + [0.0, max_translation * 0.3, 0.0],
            x0 + [0.0, 0.0, max_translation * 0.3],
        ]
    )
    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 300, "initial_simplex": initial_simplex},
    )
    theta, tx, ty = result.x

    # Safety net: Nelder-Mead isn't guaranteed to land somewhere better than where it started,
    # especially with a noisy/weak correlation signal -- explicitly compare the fit against
    # doing nothing, and fall back to the original registration (unchanged) if it isn't
    # actually better, rather than silently handing back a worse alignment.
    baseline_correlation = _match_and_correlate(moving_tree, warped0, distance_threshold, moving_values, fixed_values)
    fitted_correlation = -result.fun if -result.fun > -1e5 else None

    improved_enough = (
        fitted_correlation is not None
        and baseline_correlation is not None
        and fitted_correlation - baseline_correlation >= min_improvement
    )
    if not improved_enough:
        print(
            f"refine_alignment: the search did not find a correction that improves correlation "
            f"by at least min_improvement={min_improvement} over the original registration "
            f"(baseline r={baseline_correlation}, fitted r={fitted_correlation}) "
            "-- keeping the original registration unchanged."
        )
        refined = registration_result
        theta, tx, ty = 0.0, 0.0, 0.0
        reported_correlation = baseline_correlation
    else:
        if (
            abs(theta) >= max_rotation * 0.98
            or abs(tx) >= max_translation * 0.98
            or abs(ty) >= max_translation * 0.98
        ):
            print(
                f"refine_alignment: fitted correction (rotation={np.degrees(theta):.2f} deg, "
                f"tx={tx:.1f}, ty={ty:.1f}) is at or near its bound -- consider raising "
                "max_translation/max_rotation_deg and re-running."
            )

        correction = sitk.Euler2DTransform()
        correction.SetCenter(tuple(float(c) for c in center))
        correction.SetAngle(float(theta))
        correction.SetTranslation((float(tx), float(ty)))

        refined = RegistrationResult(
            affine_transform=registration_result.affine_transform,
            bspline_transform=registration_result.bspline_transform,
            fixed_size=registration_result.fixed_size,
            correction=correction,
        )
        reported_correlation = fitted_correlation

    merged = pipeline.align(
        moving,
        fixed,
        registration_result=refined,
        distance_threshold=distance_threshold,
        moving_table_key=moving_table_key,
        fixed_table_key=fixed_table_key,
        moving_obsm_key=moving_obsm_key,
    )
    merged.uns["spatialwarp"]["refinement"] = {
        "feature_pairs": feature_pairs,
        "rotation_deg": float(np.degrees(theta)),
        "translation": (float(tx), float(ty)),
        "correlation": reported_correlation,
        "baseline_correlation": baseline_correlation,
        "applied": refined is not registration_result,
    }
    return merged, refined


def refine_alignment_elastic(
    moving,
    fixed,
    registration_result,
    feature_pairs,
    moving_table_key=None,
    fixed_table_key=None,
    moving_obsm_key="moving_features",
    distance_threshold=20.0,
    raster_upsample=1,
    mesh_size=(8, 8),
    number_of_iterations=100,
    min_improvement=0.02,
):
    """Elastic (B-spline) variant of :func:`refine_alignment`.

    Instead of a single global rigid nudge, this fits a *local* correction: the matched
    feature(s) are rasterized into two images on a shared grid (in ``registration_result``'s
    moving-pixel-space), and :func:`spatialwarp.registration.register_elastic` is run between
    them via Mattes mutual information -- the same machinery this package normally uses to
    register two real images, just applied to feature-intensity rasters instead of pixels.

    Reach for this over :func:`refine_alignment` only once a rigid nudge visibly isn't enough
    -- e.g. one region of tissue needs a different correction than another, from a local
    fold/stretch the base registration didn't capture. It's a much heavier fit (a full B-spline
    mesh vs. 3 rigid parameters), so it can overfit noise a rigid search wouldn't.

    Parameters
    ----------
    moving, fixed : spatialdata.SpatialData
    registration_result : spatialwarp.registration.RegistrationResult
        An already-fit registration (from landmarks, elastic registration, and/or
        :func:`refine_alignment`) to further refine.
    feature_pairs : list of (str, str)
        Same meaning as in :func:`refine_alignment`. With more than one pair, each side's
        raster is the mean of its features' rasterized values (each independently scaled by
        its own max magnitude first, so features on very different scales don't dominate).
    distance_threshold : float
        Same meaning as in :func:`spatialwarp.pipeline.align` -- used for the final merge only
        (unlike :func:`refine_alignment`, this doesn't search over it).
    raster_upsample : int
        Extra resolution beyond the coarser side's own native pitch (see
        :func:`_rasterize_shared`) -- raise for a finer correction mesh, at the cost of a
        sparser raster to register from.
    mesh_size, number_of_iterations :
        Passed through to :func:`spatialwarp.registration.register_elastic`.
    min_improvement : float
        Same meaning as in :func:`refine_alignment` -- the fitted correction must beat the
        no-correction baseline by at least this much or it's discarded. Worth keeping fairly
        strict here, since a full B-spline mesh has many more ways to fit noise than a rigid
        3-parameter search does.

    Returns
    -------
    merged : anndata.AnnData
        Same shape as :func:`spatialwarp.pipeline.align`'s return, using the refined transform.
    refined_registration_result : spatialwarp.registration.RegistrationResult
        ``registration_result`` with the fitted elastic correction attached -- ``.save()``/
        ``.load()`` like any other registration.
    """
    from . import pipeline

    moving_table = extract_table(moving, moving_table_key)
    fixed_table = extract_table(fixed, fixed_table_key)

    moving_xy = moving_table.obsm["spatial"]
    fixed_xy = fixed_table.obsm["spatial"]

    moving_values = [_feature_values(moving_table, m) for m, _ in feature_pairs]
    fixed_values = [_feature_values(fixed_table, f) for _, f in feature_pairs]
    moving_tree = cKDTree(moving_xy)

    x0, y0 = registration_result.warp_points_fixed_to_moving(fixed_xy[:, 0], fixed_xy[:, 1])
    warped0 = np.column_stack([x0, y0])
    baseline_correlation = _match_and_correlate(moving_tree, warped0, distance_threshold, moving_values, fixed_values)

    def _combined_signal(table, names):
        normed = []
        for name in names:
            values = _feature_values(table, name)
            scale = np.nanmax(np.abs(values))
            normed.append(values / scale if scale > 0 else values)
        return np.mean(normed, axis=0)

    moving_signal = _combined_signal(moving_table, [m for m, _ in feature_pairs])
    fixed_signal = _combined_signal(fixed_table, [f for _, f in feature_pairs])

    # both rasters live on the same shared grid, in registration_result's moving-pixel-space
    # (warped0 is fixed's points already warped into that space by the existing registration)
    moving_raster, fixed_raster, raster_transform = _rasterize_shared(
        moving_xy, moving_signal, warped0, fixed_signal, upsample=raster_upsample
    )

    elastic = register_elastic(
        moving_image=moving_raster,
        fixed_image=fixed_raster,
        mesh_size=mesh_size,
        number_of_iterations=number_of_iterations,
    )

    # elastic.bspline_transform operates in the raster's own pixel-index space; wrap it so the
    # composed correction operates directly in moving-pixel-space physical units, matching
    # every other transform already chained in warp_points_fixed_to_moving. Note:
    # sitk.CompositeTransform applies the *last*-added transform first (verified directly --
    # not the "first added, first applied" order every other composite in this package was
    # written to assume), so these are added in reverse of their intended execution order:
    # to_pixel -> bspline -> to_physical.
    to_pixel, to_physical = _pixel_physical_transforms(raster_transform)
    correction = sitk.CompositeTransform(2)
    correction.AddTransform(to_physical)
    correction.AddTransform(elastic.bspline_transform)
    correction.AddTransform(to_pixel)

    candidate = RegistrationResult(
        affine_transform=registration_result.affine_transform,
        bspline_transform=registration_result.bspline_transform,
        fixed_size=registration_result.fixed_size,
        correction=correction,
    )

    # Safety net, same reasoning as refine_alignment: a full B-spline mesh has plenty of
    # freedom to fit noise rather than genuine local misalignment, especially with a weak
    # correlation signal -- explicitly compare against doing nothing and fall back to the
    # original registration (unchanged) if the elastic fit isn't actually better.
    x_after, y_after = candidate.warp_points_fixed_to_moving(fixed_xy[:, 0], fixed_xy[:, 1])
    fitted_correlation = _match_and_correlate(
        moving_tree, np.column_stack([x_after, y_after]), distance_threshold, moving_values, fixed_values
    )

    improved_enough = (
        fitted_correlation is not None
        and baseline_correlation is not None
        and fitted_correlation - baseline_correlation >= min_improvement
    )
    if not improved_enough:
        print(
            f"refine_alignment_elastic: the fitted correction did not improve correlation by at "
            f"least min_improvement={min_improvement} over the original registration "
            f"(baseline r={baseline_correlation}, fitted r={fitted_correlation}) "
            "-- keeping the original registration unchanged."
        )
        refined = registration_result
        reported_correlation = baseline_correlation
        applied = False
    else:
        refined = candidate
        reported_correlation = fitted_correlation
        applied = True

    merged = pipeline.align(
        moving,
        fixed,
        registration_result=refined,
        distance_threshold=distance_threshold,
        moving_table_key=moving_table_key,
        fixed_table_key=fixed_table_key,
        moving_obsm_key=moving_obsm_key,
    )
    merged.uns["spatialwarp"]["refinement"] = {
        "feature_pairs": feature_pairs,
        "kind": "elastic",
        "mesh_size": tuple(mesh_size),
        "correlation": reported_correlation,
        "baseline_correlation": baseline_correlation,
        "applied": applied,
    }
    return merged, refined
