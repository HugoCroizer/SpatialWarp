"""Minimal reader for IndicaLabs-format multi-channel pyramidal TIFFs (the export format
used by several Akoya/Opal multiplex immunofluorescence scanners) -- not a general-purpose
multiplex reader, just enough to pull one pyramid level's channels + names out, by parsing the
vendor's own ``<indica>`` XML stored in the TIFF's ``ImageDescription`` tag.

The file has no OME metadata (``tifffile``'s usual multi-page/pyramid grouping doesn't apply),
but every individual (channel, pyramid level) pair is its own plain TIFF page/IFD, and the
``<indica>`` XML says exactly which IFD holds which -- so reading a whole low-resolution level
(for landmark picking, or as the point grid to align onto Visium) is just a handful of cheap
single-page reads, never the full-resolution file.
"""

import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def parse_indica_metadata(tif_path):
    """Parse the ``<indica>`` XML in ``tif_path``'s first page.

    Returns
    -------
    ifd_by_channel_level : dict[(int, int), int]
        Maps ``(channel_id, level)`` -> the TIFF page index (IFD) holding that plane.
    channel_names : dict[int, str]
        Maps ``channel_id`` -> its human-readable name (e.g. ``"DAPI (DAPI)"``).
    """
    with tifffile.TiffFile(str(tif_path)) as tf:
        desc = tf.pages[0].tags["ImageDescription"].value
    root = ET.fromstring(desc)

    ifd_by_channel_level = {
        (int(dim.attrib["channel"]), int(dim.attrib["level"])): int(dim.attrib["ifd"])
        for dim in root.iter("dimension")
    }
    channel_names = {int(ch.attrib["id"]): ch.attrib["name"] for ch in root.iter("channel")}
    return ifd_by_channel_level, channel_names


def read_pyramid_level(tif_path, level):
    """Read every channel of one pyramid level.

    Parameters
    ----------
    tif_path : str
    level : int
        ``0`` is full resolution; higher numbers are progressively smaller precomputed
        pyramid levels (this format halves both dimensions per level).

    Returns
    -------
    stack : np.ndarray of shape (n_channels, H, W)
    channel_names : list[str]
        Same order as ``stack``'s first axis.
    """
    ifd_by_channel_level, channel_names = parse_indica_metadata(tif_path)
    channel_ids = sorted(channel_names)

    with tifffile.TiffFile(str(tif_path)) as tf:
        planes = [tf.pages[ifd_by_channel_level[(c, level)]].asarray() for c in channel_ids]

    stack = np.stack(planes, axis=0)
    names = [channel_names[c] for c in channel_ids]
    return stack, names
