"""Scoring model output against annotated ground truth.

Separate from models/ and from the validation/ package under crowd_flow:
that one checks the flow ESTIMATOR against synthetic warps (does the maths
recover a field it was given), which is a different question from whether a
model's output matches what a human saw in real footage.
"""

from evaluation.counting import (EvalResult, FrameResult, evaluate_camera,
                                 match_points)

__all__ = ["EvalResult", "FrameResult", "evaluate_camera", "match_points"]
