"""
Indian number-plate text handling: format model, character correction,
and multi-frame voting.

OCR on plates fails in a very specific, very predictable way: it confuses
characters that look alike. Reading a rendered `MH15DS7121` back through
EasyOCR in this repo's own test returned **MH1SDS7121** — the `5` became an
`S`. That is not a marginal error to shrug at; a single wrong character
makes the plate useless for lookup or matching.

The fix is that Indian plates have a rigid structure, so we know what
*kind* of character belongs at each position:

    MH    15    DS     7121
    ^^    ^^    ^^     ^^^^
    state RTO   series number
    2 ltr 2 dig 1-3ltr 4 dig

A letter appearing where a digit must be is unambiguously an OCR error,
and the correction is a lookup away. `S` in a digit slot is a `5`; `O` is
a `0`. Applying that turns MH1SDS7121 back into MH15DS7121.

Voting across frames is the second half. A vehicle is visible for many
frames, so the plate gets read many times; taking the most frequent
corrected reading is far more reliable than trusting any single frame,
where motion blur or a bad angle can corrupt one character.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# Characters OCR mixes up, keyed by what should be there.
# Applied only where the plate format says the class is certain, so a
# genuine letter in a letter slot is never rewritten.
_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6", "C": "6",
    "T": "7", "Y": "7",
    "B": "8",
}
_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Union Territory / state codes. Used to sanity-check the first two
# characters — a plate that starts with a code that doesn't exist is
# almost certainly a misread rather than a real registration.
_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "UA", "WB",
}

# LL <RTO> L(1..3) DDDD, e.g. MH15DS7121.
#
# The RTO chunk is not always two digits. Delhi encodes a vehicle category
# letter in it — DL 8C AF 5030 — and treating that slot as digits-only
# "corrected" the C into a 6, turning a valid plate into a wrong one.
# Two digits, or a digit followed by a letter.
_PLATE_RE = re.compile(r"^([A-Z]{2})(\d{2}|\d[A-Z])([A-Z]{1,3})(\d{4})$")

# Bharat-series plates: 22 BH 1234 AA
_BH_RE = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")

# States whose RTO code carries a vehicle-category letter (DL 8C AF 5030).
# Deliberately narrow: every state added here makes one more OCR letter
# un-correctable in that slot.
_LETTER_RTO_STATES = {"DL"}


def normalize_raw(text: str) -> str:
    """Strip everything OCR adds that a plate never contains."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _coerce(chunk: str, want_digits: bool) -> str:
    table = _TO_DIGIT if want_digits else _TO_LETTER
    return "".join(table.get(c, c) for c in chunk)


def correct_format(raw: str) -> Optional[str]:
    """Apply position-aware character correction to a raw OCR string.

    Returns the corrected plate, or None if the string can't be made to fit
    any known Indian plate layout — which is the useful signal that this
    was a misdetection (a bumper sticker, a bit of trim) rather than a
    plate we simply misread.
    """
    s = normalize_raw(raw)
    if len(s) < 8 or len(s) > 11:
        return None

    candidates: list[str] = []

    # Standard layout: 4-digit tail, 1-3 letter series, LL + RTO head.
    # Both the series length and the RTO shape (two digits, or Delhi's
    # digit+letter as in DL 8C) are ambiguous from a bare string, so every
    # combination is generated and scored below rather than returning the
    # first that happens to parse.
    for series_len in (2, 3, 1):
        if len(s) != 4 + series_len + 4:
            continue
        state = _coerce(s[0:2], want_digits=False)
        if state not in _STATE_CODES:
            continue
        series = _coerce(s[4:4 + series_len], want_digits=False)
        number = _coerce(s[4 + series_len:], want_digits=True)

        rto_options = [_coerce(s[2:4], want_digits=True)]     # "15"
        if state in _LETTER_RTO_STATES:
            # Only offered where it's real. Allowing digit+letter RTO
            # everywhere makes "MH1S..." parse as RTO "1S" with zero edits,
            # so the least-edits rule below would keep the OCR's S instead
            # of correcting it to the 5 it actually is.
            rto_options.append(_coerce(s[2:3], True) + _coerce(s[3:4], False))

        for rto in rto_options:
            candidate = f"{state}{rto}{series}{number}"
            if _PLATE_RE.match(candidate):
                candidates.append(candidate)

    # Bharat series: DD BH DDDD LL
    if len(s) in (9, 10):
        candidate = (_coerce(s[0:2], True) + _coerce(s[2:4], False)
                     + _coerce(s[4:8], True) + _coerce(s[8:], False))
        if _BH_RE.match(candidate):
            candidates.append(candidate)

    if not candidates:
        return None

    # Fewest edits wins. Reading DL8CAF5030 as two RTO digits "corrects" the
    # C into a 6 and yields a technically valid DL86AF5030 — parseable, and
    # wrong. Preferring the interpretation that leaves the OCR output alone
    # keeps the real plate, since every rewritten character is a guess.
    def edits(candidate: str) -> int:
        return sum(1 for a, b in zip(s, candidate) if a != b)

    return min(candidates, key=edits)


def is_valid(plate: str) -> bool:
    if not plate:
        return False
    if _BH_RE.match(plate):
        return True
    m = _PLATE_RE.match(plate)
    return bool(m) and m.group(1) in _STATE_CODES


def format_display(plate: str) -> str:
    """MH15DS7121 -> 'MH 15 DS 7121' for the gallery caption."""
    m = _PLATE_RE.match(plate or "")
    if m:
        return " ".join(m.groups())
    m = _BH_RE.match(plate or "")
    if m:
        return " ".join(m.groups())
    return plate or ""


@dataclass
class PlateVote:
    """Accumulates plate readings for one tracked vehicle.

    Confidence-weighted rather than a plain count: a clear read from a
    close-up frame should outweigh several blurry ones from a distance.
    Readings that survive format correction are weighted far more heavily
    than those that don't, since fitting the layout is strong evidence the
    read was real.
    """
    readings: Counter = field(default_factory=Counter)
    raw_readings: Counter = field(default_factory=Counter)
    best_conf: float = 0.0
    n_reads: int = 0

    def add(self, raw_text: str, ocr_conf: float) -> Optional[str]:
        raw = normalize_raw(raw_text)
        if not raw:
            return None
        self.n_reads += 1
        self.raw_readings[raw] += 1

        corrected = correct_format(raw)
        if corrected:
            # Valid-format reads dominate: a string that snaps onto the
            # plate layout is far more likely to be a real plate than one
            # that doesn't, regardless of raw OCR confidence.
            self.readings[corrected] += 1.0 + ocr_conf
            self.best_conf = max(self.best_conf, ocr_conf)
        return corrected

    def result(self) -> tuple[Optional[str], float]:
        """Most likely plate for this vehicle, and a 0-1 agreement score."""
        if not self.readings:
            return None, 0.0
        plate, weight = self.readings.most_common(1)[0]
        total = sum(self.readings.values())
        agreement = weight / total if total else 0.0
        return plate, agreement
