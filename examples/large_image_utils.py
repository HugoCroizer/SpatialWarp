"""Read a small, downsampled view of a very large (whole-slide-scale) TIFF without
materializing the full image in memory.

Some H&E scans are tens of thousands of pixels per side (multi-GB, uncompressed,
tiled TIFFs). Reading them via a normal ``imageio``/``tifffile.imread()`` call means
decoding the entire file; computing a lower level of a SpatialData multiscale image
pyramid is no better, since those levels are lazily coarsened from the full-resolution
dask array and still end up reading (and averaging) every source pixel. A zarr-backed
strided read only touches the tiles actually needed for the requested stride -- plenty
for landmark picking and elastic registration, neither of which needs full resolution.
"""

import numpy as np
import tifffile
import zarr


def load_downsampled_image(path, downsample):
    """Read every ``downsample``-th pixel of the image at ``path``.

    Parameters
    ----------
    path : str
    downsample : int
        Stride in both dimensions, e.g. ``20`` keeps 1 out of every 20 pixels
        per axis.

    Returns
    -------
    image : np.ndarray of shape (H, W, C)
    scale_factor : float
        ``image``'s size relative to the original (``1 / downsample``,
        roughly) -- multiply any point coordinate in the original image's
        pixel space by this to land in ``image``'s pixel space.
    """
    store = tifffile.imread(str(path), aszarr=True)
    z = zarr.open(store, mode="r")
    full_height = z.shape[0]
    image = np.asarray(z[::downsample, ::downsample, ...])
    scale_factor = image.shape[0] / full_height
    return image, scale_factor
