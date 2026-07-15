"""Cross-modality elastic image registration via SimpleITK.

Landmark-guided affine initialization followed by intensity-based B-spline
elastic registration. Pure Python — no external tools (Fiji/BUnwarpJ) needed.
"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


def _to_sitk_image(array):
    """Convert a (H,W) or (H,W,C) numpy array into a 2D grayscale float32
    sitk.Image with identity spacing/origin, so physical coordinates equal
    pixel (x, y) coordinates."""
    array = np.asarray(array)
    if array.ndim == 3:
        array = array.mean(axis=2)
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((1.0, 1.0))
    image.SetOrigin((0.0, 0.0))
    return image


def _flatten_landmarks(xy):
    """(N, 2) array -> flat [x1, y1, x2, y2, ...] list expected by
    LandmarkBasedTransformInitializer."""
    xy = np.asarray(xy, dtype=float)
    return xy.reshape(-1).tolist()


@dataclass
class RegistrationResult:
    affine_transform: sitk.Transform
    bspline_transform: sitk.Transform
    fixed_size: tuple

    def warp_points_fixed_to_moving(self, x, y):
        """Map point coordinates from fixed-image pixel space into
        moving-image pixel space."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_out = np.empty_like(x)
        y_out = np.empty_like(y)
        for i in range(len(x)):
            p = self.bspline_transform.TransformPoint((float(x[i]), float(y[i])))
            p = self.affine_transform.TransformPoint(p)
            x_out[i], y_out[i] = p
        return x_out, y_out

    def _combined_transform(self):
        combined = sitk.CompositeTransform(2)
        combined.AddTransform(self.bspline_transform)
        combined.AddTransform(self.affine_transform)
        return combined

    def warp_image(self, moving_image):
        """Warp `moving_image` (numpy array) onto the fixed image's pixel grid."""
        moving_sitk = _to_sitk_image(moving_image)
        width, height = self.fixed_size
        fixed_sitk = sitk.Image(int(width), int(height), sitk.sitkFloat32)
        fixed_sitk.SetSpacing((1.0, 1.0))
        fixed_sitk.SetOrigin((0.0, 0.0))

        warped = sitk.Resample(
            moving_sitk, fixed_sitk, self._combined_transform(), sitk.sitkLinear, 0.0
        )
        return sitk.GetArrayFromImage(warped)

    def save(self, path):
        sitk.WriteTransform(self.affine_transform, str(path) + ".affine.tfm")
        sitk.WriteTransform(self.bspline_transform, str(path) + ".bspline.tfm")

    @classmethod
    def load(cls, path, fixed_size):
        affine = sitk.ReadTransform(str(path) + ".affine.tfm")
        bspline = sitk.ReadTransform(str(path) + ".bspline.tfm")
        return cls(affine_transform=affine, bspline_transform=bspline, fixed_size=fixed_size)


def register_elastic(
    moving_image,
    fixed_image,
    moving_landmarks=None,
    fixed_landmarks=None,
    mesh_size=(8, 8),
    number_of_iterations=100,
    sampling_percentage=0.2,
):
    """Register ``moving_image`` to ``fixed_image``.

    Landmark-guided affine initialization (if landmarks are given) followed by
    intensity-based elastic B-spline refinement.

    Parameters
    ----------
    moving_image, fixed_image : np.ndarray
    moving_landmarks, fixed_landmarks : array-like of shape (N, 2), or None
        Corresponding (x, y) landmark points, e.g. from
        :func:`spatialwarp.landmark_picker.pick_landmarks`.
    mesh_size : tuple of int
        BSpline control point grid size (more control points = more local
        flexibility, but needs more data/iterations to fit well).

    Returns
    -------
    RegistrationResult
    """
    fixed_sitk = _to_sitk_image(fixed_image)
    moving_sitk = _to_sitk_image(moving_image)

    affine = sitk.AffineTransform(2)
    has_landmarks = (
        fixed_landmarks is not None
        and moving_landmarks is not None
        and len(fixed_landmarks) > 0
        and len(moving_landmarks) > 0
    )
    if has_landmarks:
        affine = sitk.LandmarkBasedTransformInitializer(
            affine,
            _flatten_landmarks(fixed_landmarks),
            _flatten_landmarks(moving_landmarks),
        )

    bspline = sitk.BSplineTransformInitializer(fixed_sitk, list(mesh_size))

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(sampling_percentage)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=number_of_iterations,
    )
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(bspline, inPlace=True)
    R.SetMovingInitialTransform(affine)

    optimized_bspline = R.Execute(fixed_sitk, moving_sitk)

    return RegistrationResult(
        affine_transform=affine,
        bspline_transform=optimized_bspline,
        fixed_size=fixed_sitk.GetSize(),
    )
