"""
Annotation-free validation for the dense optical flow module.

Three routes, each manufacturing ground truth from a different source, chosen
so that each one covers the others' blind spots:

  (a) synthetic_warp  exact truth, real imagery, synthetic motion, any density
  (b) cross_camera    error bound, real motion, real density, needs 2 cameras
  (c) cross_family    approximate truth, real motion, low density only

                        exact error?   real motion?   high density?
    (a) synthetic warp      yes             no             yes
    (b) cross-camera      bound only        yes            yes
    (c) cross-family      approximate       yes            no

Only (b) is both real-motion and high-density, which makes it the
load-bearing route for crush conditions — and the one most dependent on
calibration quality.

None of the three validates whether the DERIVED metrics (divergence,
counterflow, turbulence) predict crush risk.  They validate that the velocity
field is measured correctly.  Whether a divergence of -1.5 per cell actually
means danger is a separate question that no amount of flow validation
answers.
"""

from models.crowd_flow.validation.report import (
    Measurement,
    RouteResult,
    ValidationReport,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_SKIPPED,
    STATUS_ERROR,
)
from models.crowd_flow.validation.synthetic_warp import (
    SyntheticWarpValidator,
    WarpCase,
    default_cases,
    warp_frame,
    endpoint_error,
    translation_field,
    radial_field,
    shear_field,
    rotation_field,
)
from models.crowd_flow.validation.cross_camera import (
    CrossCameraValidator,
    CameraView,
)
from models.crowd_flow.validation.cross_family import CrossFamilyValidator

__all__ = [
    "Measurement", "RouteResult", "ValidationReport",
    "STATUS_PASS", "STATUS_FAIL", "STATUS_SKIPPED", "STATUS_ERROR",
    "SyntheticWarpValidator", "WarpCase", "default_cases",
    "warp_frame", "endpoint_error",
    "translation_field", "radial_field", "shear_field", "rotation_field",
    "CrossCameraValidator", "CameraView",
    "CrossFamilyValidator",
]
