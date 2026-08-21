"""
Shared result types for the annotation-free validation routes.

Every route answers the same question — "how wrong is the flow estimator?" —
from a different source of truth, and each one is blind to something the
others can see.  The report format therefore carries, alongside the numbers,
an explicit ``caveat`` per route describing what that route CANNOT tell you.

This is not decoration.  Before the work in this module, the project's
synthetic suite reported 11/11 passing while global motion compensation was
corrupting the flow field by up to 179 px/frame on real footage — because
synthetic tests contain no moving objects for the GMC estimator to lock onto.
Green results are a necessary condition for trusting the estimator, never a
sufficient one, and the report is structured so a reader cannot see the
former without also seeing the latter.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Route status values.
STATUS_PASS    = "pass"
STATUS_FAIL    = "fail"
STATUS_SKIPPED = "skipped"   # preconditions absent (e.g. no camera pair)
STATUS_ERROR   = "error"     # the route itself failed to run


@dataclass
class Measurement:
    """
    One reported number.

    ``tolerance`` is the threshold the value is judged against; None means the
    measurement is informational and does not affect pass/fail.  By default a
    measurement passes when value <= tolerance (errors: smaller is better);
    set higher_is_better for scores where the test is value >= tolerance.
    """
    label: str
    value: float
    units: str
    tolerance: Optional[float] = None
    higher_is_better: bool = False
    note: str = ""

    @property
    def passed(self) -> Optional[bool]:
        if self.tolerance is None:
            return None
        if self.higher_is_better:
            return self.value >= self.tolerance
        return self.value <= self.tolerance


@dataclass
class RouteResult:
    """Outcome of one validation route."""
    route: str                       # machine key, e.g. "synthetic_warp"
    title: str                       # human label
    status: str                      # STATUS_*
    summary: str                     # one-line verdict
    caveat: str                      # what this route cannot tell you
    measurements: list[Measurement] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def skipped(cls, route: str, title: str, why: str, caveat: str) -> "RouteResult":
        return cls(route=route, title=title, status=STATUS_SKIPPED,
                   summary=why, caveat=caveat)

    @classmethod
    def errored(cls, route: str, title: str, exc: Exception,
                caveat: str = "") -> "RouteResult":
        return cls(route=route, title=title, status=STATUS_ERROR,
                   summary=f"{exc.__class__.__name__}: {exc}", caveat=caveat)

    def resolve_status(self) -> None:
        """Set status from the measurements that carry a tolerance."""
        judged = [m for m in self.measurements if m.tolerance is not None]
        if not judged:
            self.status = STATUS_PASS
            return
        self.status = STATUS_PASS if all(m.passed for m in judged) else STATUS_FAIL

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict() drops the `passed` property, which the UI needs.
        for m_dict, m in zip(d["measurements"], self.measurements):
            m_dict["passed"] = m.passed
        return d


@dataclass
class ValidationReport:
    """A full run of one or more routes."""
    routes: list[RouteResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    source: str = ""                 # video / dataset the run was made against
    notes: str = ""

    @property
    def status(self) -> str:
        """
        Worst status across the routes that actually ran.

        A skipped route does not make the report fail — it makes it
        incomplete, which the UI shows separately.  Rolling "skipped" into
        "pass" would let an unrun route read as a satisfied one.
        """
        ran = [r.status for r in self.routes if r.status != STATUS_SKIPPED]
        if not ran:
            return STATUS_SKIPPED
        if STATUS_ERROR in ran:
            return STATUS_ERROR
        return STATUS_FAIL if STATUS_FAIL in ran else STATUS_PASS

    def to_dict(self) -> dict:
        return {
            "status":     self.status,
            "created_at": self.created_at,
            "source":     self.source,
            "notes":      self.notes,
            "routes":     [r.to_dict() for r in self.routes],
        }

    def write_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
