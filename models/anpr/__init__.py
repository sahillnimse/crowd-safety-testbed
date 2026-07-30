"""ANPR: vehicle capture with number-plate recognition."""

from models.anpr.anpr import ANPRDetector
from models.anpr.indian_anpr import IndianANPRDetector
from models.anpr.rapid_ocr_wrapper import RapidOCRDetector
from models.anpr._rapid_ocr import PlateRapidOCR
from models.anpr._plate_text import (
    correct_format,
    format_display,
    is_valid,
    normalize_raw,
)

__all__ = [
    "ANPRDetector",
    "IndianANPRDetector",
    "RapidOCRDetector",
    "PlateRapidOCR",
    "correct_format",
    "format_display",
    "is_valid",
    "normalize_raw",
]
