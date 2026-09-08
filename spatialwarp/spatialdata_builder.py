"""Package raw spatial data (an image + point coordinates + features) into a
SpatialData object, and extract it back out.

This is pure data conversion — no landmark picking or registration here, just
the SpatialData construction/extraction step, so any technology's loader can
produce (or consume) a ready-to-align object without pulling in any alignment
code.
"""

import numpy as np
import pandas as pd
import anndata
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, TableModel


def build_spatialdata(image, points_xy, values=None, image_key="image", table_key="table"):
    """Package ``image`` + ``points_xy`` (+ optional ``values``) into a
    SpatialData object with one image element and one table element whose
    ``obsm['spatial']`` holds ``points_xy``.

    Parameters
    ----------
    image : np.ndarray
        Reference image, ``(H, W)`` or ``(H, W, C)``.
    points_xy : array-like of shape (n, 2)
        Point coordinates in ``image``'s own pixel space — e.g. the output of
        :func:`spatialwarp.registration.RegistrationResult.warp_points_fixed_to_moving`
        or :func:`spatialwarp.grid_align.run_grid_alignment`.
    values : pandas.DataFrame, array-like, or None
        Per-point feature matrix, becomes the table's ``.X`` with ``.var_names``
        taken from the DataFrame's columns (or ``feature_0``, ``feature_1``,
        ... for a plain array). If None, the table has zero variables (points
        only).
    image_key, table_key : str
        Element keys for the returned SpatialData object.

    Returns
    -------
    spatialdata.SpatialData
    """
    points_xy = np.asarray(points_xy, dtype=float)

    if values is None:
        X = np.zeros((len(points_xy), 0), dtype=np.float32)
        var = pd.DataFrame(index=[])
    elif isinstance(values, pd.DataFrame):
        X = values.values.astype(np.float32)
        var = pd.DataFrame(index=values.columns)
    else:
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        X = values.astype(np.float32)
        var = pd.DataFrame(index=[f"feature_{i}" for i in range(values.shape[1])])

    adata = anndata.AnnData(X=X, var=var)
    adata.obsm["spatial"] = points_xy

    table = TableModel.parse(adata)
    image_element = Image2DModel.parse(np.moveaxis(np.atleast_3d(image), -1, 0))

    return SpatialData(images={image_key: image_element}, tables={table_key: table})


def _to_numpy_image(element, scale=None):
    data = getattr(element, "data", None)
    if data is None:
        # multiscale (DataTree) image: pick one pyramid level (full resolution, "scale0",
        # unless a lower one is requested -- useful to avoid ever materializing a huge
        # full-resolution whole-slide image just to get a reasonably-sized array)
        scale_name = scale if isinstance(scale, str) else f"scale{0 if scale is None else scale}"
        data = element[scale_name]["image"].data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def extract_image(sdata, key=None, scale=None):
    """Pull one image element out of a SpatialData object as a plain
    ``(H, W)`` or ``(H, W, C)`` numpy array.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
    key : str or None
        Which image element to use. Required only if ``sdata`` has more than
        one.
    scale : int, str, or None
        For a multiscale (DataTree) image, which pyramid level to extract
        (e.g. ``2`` or ``"scale2"``). ``None`` (default) uses the
        full-resolution level. Ignored for a single-resolution image.
    """
    images = sdata.images
    if not images:
        raise ValueError("SpatialData has no image elements")
    if key is None:
        if len(images) != 1:
            raise ValueError(f"SpatialData has multiple image elements {list(images)}; specify the key")
        key = next(iter(images))

    arr = _to_numpy_image(images[key], scale=scale)
    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)  # (c, y, x) -> (y, x, c)
    return arr


def extract_table(sdata, key=None):
    """Pull one table element out of a SpatialData object.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
    key : str or None
        Which table element to use. Required only if ``sdata`` has more than
        one.

    Returns
    -------
    anndata.AnnData
        The table, guaranteed to have ``obsm['spatial']``.
    """
    tables = sdata.tables
    if not tables:
        raise ValueError("SpatialData has no table elements")
    if key is None:
        if len(tables) != 1:
            raise ValueError(f"SpatialData has multiple table elements {list(tables)}; specify the key")
        key = next(iter(tables))

    table = tables[key]
    if "spatial" not in table.obsm:
        raise ValueError(f"Table '{key}' has no obsm['spatial']; pipeline requires point coordinates there")
    return table


def table_to_frame(table):
    """Return ``table``'s ``.X`` as a DataFrame with ``.var_names`` as
    columns, or None if it has zero variables (points-only table)."""
    if table.n_vars == 0:
        return None
    return pd.DataFrame(np.asarray(table.X), columns=table.var_names, index=table.obs_names)
