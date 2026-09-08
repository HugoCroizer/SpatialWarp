"""Load classic (non-HD) Visium Space Ranger output into a plain image+points
SpatialData object.

Space Ranger runs often don't save a full-resolution H&E image, only the
downscaled hires/lowres preview images -- but spot coordinates are always
recorded in full-resolution pixel space, so they don't natively match either
preview image's pixel grid. This rescales the coordinates into the hires
image's own pixel space using Space Ranger's own scale factor, then
repackages the result as a plain image + points SpatialData object, the same
shape every other object in this pipeline has.
"""

import numpy as np
import spatialdata as sd
import spatialdata_io
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

import spatialwarp as sw


def load_classic_visium_hires(path, dataset_id, counts_file=None, image_key="he", table_key="visium"):
    """Read a Space Ranger ``outs/`` folder and return ``(sdata, hires_image)``.

    Parameters
    ----------
    path : str
        Path to the Space Ranger ``outs/`` directory (or wherever the counts
        file and ``spatial/`` folder live).
    dataset_id : str
        Used to name the resulting image/shapes elements, e.g. ``"ST48"``.
    counts_file : str or None
        Pass this when the counts file isn't named the SpaceRanger default
        ``filtered_feature_bc_matrix.h5`` (e.g. a dataset-id-prefixed name).
    image_key, table_key : str
        Element keys for the returned SpatialData object.

    Returns
    -------
    sdata : spatialdata.SpatialData
        One image element (the hires H&E preview) and one table element
        whose ``obsm['spatial']`` matches that image's pixel space.
    hires_image : np.ndarray
        The same image as a plain array, handy for landmark picking / QC
        plots.
    """
    kwargs = {"counts_file": counts_file} if counts_file is not None else {}
    raw_sdata = spatialdata_io.visium(path, dataset_id=dataset_id, **kwargs)

    hires_key = f"{dataset_id}_hires_image"
    hires_scale = sd.transformations.get_transformation(
        raw_sdata.shapes[dataset_id], "downscaled_hires"
    ).scale[0]

    hires_image = sw.extract_image(raw_sdata, key=hires_key)
    table = raw_sdata.tables["table"].copy()
    table.obsm["spatial"] = table.obsm["spatial"] * hires_scale

    sdata = SpatialData(
        images={image_key: Image2DModel.parse(np.moveaxis(np.atleast_3d(hires_image), -1, 0))},
        tables={table_key: table},
    )
    return sdata, hires_image
