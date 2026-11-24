"""Versioned reports, editor confirmations, and transactional recomputation.

Semantics
---------
* Every analysis holds an ordered list of immutable report versions.
* An editor submits confirmations together with the report version they are
  looking at.  The service recomputes from the *full* confirmation set
  (previously confirmed state merged with this request's edits) and the new
  report version and the confirmations become visible together, atomically.
* A stale ``expected_version`` or an infeasible recomputation rejects the whole
  request; the prior confirmations and reports remain unchanged and queryable.
* A confirmation for a key may be *repeated* with the same value (idempotent)
  or changed, which is the editor's correction; both are accepted.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import solver


class ApiError(Exception):
    """Domain error carrying an HTTP status, a stable code, and optional detail."""

    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class AnalysisInput:
    key_names: list[str]
    templates: list[list[float]]
    band_names: list[str]
    band_energy_1: list[float]
    band_error_1: list[float]
    band_energy_2: list[float]
    band_error_2: list[float]
    gain_lower: list[float]
    gain_upper: list[float]
    time_limit: float = 10.0

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "templates": np.asarray(self.templates, dtype=float),
            "b1": np.asarray(self.band_energy_1, dtype=float),
            "e1": np.asarray(self.band_error_1, dtype=float),
            "b2": np.asarray(self.band_energy_2, dtype=float),
            "e2": np.asarray(self.band_error_2, dtype=float),
            "lo": np.asarray(self.gain_lower, dtype=float),
            "hi": np.asarray(self.gain_upper, dtype=float),
        }

    def segment_arrays(self, segment: int) -> tuple[np.ndarray, np.ndarray]:
        """Current (committed) energy/error vectors for one segment."""
        if segment == 0:
            return (
                np.asarray(self.band_energy_1, dtype=float),
                np.asarray(self.band_error_1, dtype=float),
            )
        return (
            np.asarray(self.band_energy_2, dtype=float),
            np.asarray(self.band_error_2, dtype=float),
        )

    def with_segment_readings(
        self,
        segment: int,
        energy: list[float],
        error: list[float],
    ) -> "AnalysisInput":
        """Return a copy where one segment carries replacement readings."""
        if segment == 0:
            return replace(
                self, band_energy_1=list(energy), band_error_1=list(error)
            )
        return replace(
            self, band_energy_2=list(energy), band_error_2=list(error)
        )

    def readings_view(self) -> dict[str, dict[str, list[float]]]:
        """Serializable per-segment readings held by this payload."""
        return {
            "0": {"energy": list(self.band_energy_1), "error": list(self.band_error_1)},
            "1": {"energy": list(self.band_energy_2), "error": list(self.band_error_2)},
        }


@dataclass(frozen=True)
class TakeView:
    pressed_keys: list[str]
    gains: dict[str, float]


@dataclass(frozen=True)
class ReportVersion:
    version: int
    kind: str
    # One entry per reported hypothesis ("same"/"different"), each a pair of takes.
    possibilities: list[tuple[str, list[TakeView]]]
    evidence: dict[str, Any]
    contributions: dict[int, dict[int, bool]]  # segment -> key index -> pressed
    # Per-segment band readings (energy + absolute error) this version was
    # computed from; every version keeps its own snapshot for later retrieval.
    readings: dict[str, dict[str, list[float]]]
    created_at: str


@dataclass
class Analysis:
    analysis_id: str
    payload: AnalysisInput
    reports: list[ReportVersion] = field(default_factory=list)
    confirmations: dict[int, dict[int, bool]] = field(default_factory=dict)

    @property
    def current(self) -> ReportVersion:
        return self.reports[-1]


class Store:
    """Thread-safe in-memory store; deliberately no external database."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._analyses: dict[str, Analysis] = {}

    # -- input validation -------------------------------------------------
    @staticmethod
    def _validate(payload: AnalysisInput) -> None:
        n = len(payload.key_names)
        m = len(payload.band_names)
        if n == 0:
            raise ApiError("invalid_input", "key_names must not be empty")
        if n > 24:
            raise ApiError("invalid_input", "at most 24 candidate keys are supported")
        if m == 0:
            raise ApiError("invalid_input", "band_names must not be empty")
        if len(set(payload.key_names)) != n:
            raise ApiError("invalid_input", "key_names must be unique")
        if len(set(payload.band_names)) != m:
            raise ApiError("invalid_input", "band_names must be unique")
        if len(payload.templates) != n:
            raise ApiError("invalid_input", f"templates must have {n} rows (one per key)")
        if len(payload.gain_lower) != n or len(payload.gain_upper) != n:
            raise ApiError("invalid_input", "gain_lower and gain_upper must have one entry per key")
        for label, band in (
            ("band_energy_1", payload.band_energy_1),
            ("band_error_1", payload.band_error_1),
            ("band_energy_2", payload.band_energy_2),
            ("band_error_2", payload.band_error_2),
        ):
            if len(band) != m:
                raise ApiError("invalid_input", f"{label} must have {m} entries")
        for i, row in enumerate(payload.templates):
            if len(row) != m:
                raise ApiError("invalid_input", f"templates row {i} must have {m} entries")
        if payload.time_limit <= 0:
            raise ApiError("invalid_input", "time_limit must be positive")

        arrays = payload.arrays()
        try:
            solver._validate_templates(arrays["templates"])
            solver._validate_energy("band_energy_1", arrays["b1"], m)
            solver._validate_energy("band_energy_2", arrays["b2"], m)
            solver._validate_error("band_error_1", arrays["e1"], m)
            solver._validate_error("band_error_2", arrays["e2"], m)
            solver._validate_gains(arrays["lo"], arrays["hi"], n)
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc

    # -- rendering --------------------------------------------------------
    @staticmethod
    def _take_views(payload: AnalysisInput, takes: tuple[solver.Solution, solver.Solution]) -> list[TakeView]:
        return [
            TakeView(
                pressed_keys=[payload.key_names[k] for k in sol.pressed],
                gains={payload.key_names[k]: float(sol.gains[k]) for k in sol.pressed},
            )
            for sol in takes
        ]

    def _version_from_verdict(
        self,
        payload: AnalysisInput,
        verdict: solver.Verdict,
        contributions: dict[int, dict[int, bool]],
        version: int,
    ) -> ReportVersion:
        possibilities = [
            (name, self._take_views(payload, takes))
            for name, takes in verdict.possibilities.items()
        ]
        return ReportVersion(
            version=version,
            kind=verdict.kind,
            possibilities=possibilities,
            evidence=verdict.evidence,
            contributions={seg: dict(keys) for seg, keys in contributions.items()},
            readings=payload.readings_view(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # -- public API -------------------------------------------------------
    def create(self, payload: AnalysisInput) -> Analysis:
        self._validate(payload)
        arrays = payload.arrays()
        try:
            verdict = solver.classify(
                arrays["templates"],
                (arrays["b1"], arrays["b2"]),
                (arrays["e1"], arrays["e2"]),
                arrays["lo"],
                arrays["hi"],
                confirmations={},
                time_limit=payload.time_limit,
            )
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc
        except solver.SolverInconclusive as exc:
            raise ApiError("solver_inconclusive", str(exc), status=503) from exc

        analysis = Analysis(analysis_id=str(uuid.uuid4()), payload=payload)
        report = self._version_from_verdict(payload, verdict, {}, version=1)
        with self._lock:
            analysis.reports.append(report)
            self._analyses[analysis.analysis_id] = analysis
            return analysis

    def get(self, analysis_id: str) -> Analysis:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis is None:
                raise ApiError("not_found", "analysis not found", status=404)
            return analysis

    def snapshot(self, analysis_id: str, version: int | None = None) -> tuple[Analysis, ReportVersion]:
        """Return the analysis and one immutable report, read under the lock."""
        with self._lock:
            analysis = self.get(analysis_id)
            if version is None:
                return analysis, analysis.current
            for report in analysis.reports:
                if report.version == version:
                    return analysis, report
            raise ApiError(
                "not_found",
                f"version {version} not found for analysis {analysis_id}",
                status=404,
                detail={"available_versions": [r.version for r in analysis.reports]},
            )

    def version_index(self, analysis_id: str) -> tuple[int, list[tuple[int, str, str]]]:
        with self._lock:
            analysis = self.get(analysis_id)
            return analysis.current.version, [
                (r.version, r.kind, r.created_at) for r in analysis.reports
            ]

    def confirmation_snapshot(self, analysis_id: str) -> tuple[Analysis, int, dict[int, dict[int, bool]]]:
        with self._lock:
            analysis = self.get(analysis_id)
            return (
                analysis,
                analysis.current.version,
                {seg: dict(keys) for seg, keys in analysis.confirmations.items()},
            )

    def confirm(
        self,
        analysis_id: str,
        expected_version: int,
        incoming: dict[int, dict[int, bool]],
    ) -> Analysis:
        """Apply confirmations atomically against an expected report version.

        Rejected wholesale (prior state untouched) when the version is stale or
        recomputation from the merged confirmation set has no solution.  The
        solver runs outside the lock; a version check inside a second locked
        section guards the commit so two concurrent editors cannot both advance
        the same version.
        """
        with self._lock:
            analysis = self.get(analysis_id)
            current = analysis.current
            if not incoming:
                raise ApiError("invalid_input", "at least one confirmation is required")

            n = len(analysis.payload.key_names)
            for seg, keys in incoming.items():
                if seg not in (0, 1):
                    raise ApiError("invalid_input", f"segment {seg} must be 0 or 1")
                for key in keys:
                    if not 0 <= key < n:
                        raise ApiError("invalid_input", f"key index {key} out of range for {n} keys")

            if expected_version != current.version:
                raise ApiError(
                    "stale_version",
                    f"report has advanced to version {current.version}; request targeted "
                    f"version {expected_version}",
                    status=409,
                    detail={"current_version": current.version, "expected_version": expected_version},
                )

            # Snapshot state and merge: this request may repeat or correct
            # earlier confirmations.
            snapshot = {seg: dict(keys) for seg, keys in analysis.confirmations.items()}
            observed_version = current.version
            merged = {seg: dict(keys) for seg, keys in snapshot.items()}
            for seg, keys in incoming.items():
                target = merged.setdefault(seg, {})
                target.update(keys)

        flat: solver.Confirmations = {
            (seg, key): pressed for seg, keys in merged.items() for key, pressed in keys.items()
        }
        arrays = analysis.payload.arrays()
        try:
            verdict = solver.classify(
                arrays["templates"],
                (arrays["b1"], arrays["b2"]),
                (arrays["e1"], arrays["e2"]),
                arrays["lo"],
                arrays["hi"],
                confirmations=flat,
                time_limit=analysis.payload.time_limit,
            )
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc
        except solver.SolverInconclusive as exc:
            raise ApiError("solver_inconclusive", str(exc), status=503) from exc

        if verdict.kind == "impossible":
            raise ApiError(
                "infeasible_after_confirmations",
                "no reconstruction fits both takes under the confirmed edits; "
                "the confirmation was rolled back",
                status=409,
                detail={
                    "rejected_confirmations": {
                        str(seg): {analysis.payload.key_names[k]: v for k, v in keys.items()}
                        for seg, keys in incoming.items()
                    },
                    "evidence": verdict.evidence,
                    "current_version": observed_version,
                    "current_confirmations": {
                        str(seg): {analysis.payload.key_names[k]: v for k, v in keys.items()}
                        for seg, keys in snapshot.items()
                    },
                },
            )

        report = self._version_from_verdict(analysis.payload, verdict, merged, observed_version + 1)
        with self._lock:
            # Re-check under the lock: another editor may have advanced the report
            # while the solver was running.  Reject rather than append out of order.
            if analysis.current.version != observed_version:
                raise ApiError(
                    "stale_version",
                    f"report advanced to version {analysis.current.version} during recomputation",
                    status=409,
                    detail={
                        "current_version": analysis.current.version,
                        "expected_version": expected_version,
                    },
                )
            analysis.confirmations = merged
            analysis.reports.append(report)
            return analysis

    # -- segment reading replacement --------------------------------------
    def replace_readings(
        self,
        analysis_id: str,
        expected_version: int,
        segment: int,
        energy: list[float],
        error: list[float],
    ) -> Analysis:
        """Atomically replace one segment's band readings and recompute.

        The submitted readings are accepted only when a reconstruction still
        exists for both segments: then the new readings and the new report
        version take effect together, while all editor confirmations carry
        over unchanged.  Otherwise nothing is applied and the response carries
        a minimum-cardinality retest list with a witness for the segment.

        As with confirmations, the solver runs outside the store lock; reads
        and confirmation submissions stay available during recomputation and
        a version change before commit rejects the request (the editor must
        resubmit against the new version).
        """
        with self._lock:
            analysis = self.get(analysis_id)
            current = analysis.current
            payload = analysis.payload
            if segment not in (0, 1):
                raise ApiError("invalid_input", f"segment {segment} must be 0 or 1")
            m = len(payload.band_names)
            if len(energy) != m or len(error) != m:
                raise ApiError(
                    "invalid_input",
                    f"segment readings must each have {m} entries (one per band)",
                )
            if expected_version != current.version:
                raise ApiError(
                    "stale_version",
                    f"report has advanced to version {current.version}; request targeted "
                    f"version {expected_version}",
                    status=409,
                    detail={"current_version": current.version, "expected_version": expected_version},
                )
            # Only reports that currently admit a feasible reconstruction may
            # be re-read; an already-impossible analysis must stay untouched.
            if current.kind == "impossible":
                raise ApiError(
                    "no_feasible_reconstruction",
                    "the current report has no feasible reconstruction; new readings "
                    "are only accepted against a report that can currently be reconstructed",
                    status=409,
                    detail={"current_version": current.version},
                )

            observed_version = current.version
            committed_payload = payload
            confirmations_snapshot = {
                seg: dict(keys) for seg, keys in analysis.confirmations.items()
            }

        b_new = np.asarray(energy, dtype=float)
        e_new = np.asarray(error, dtype=float)
        b_old, e_old = committed_payload.segment_arrays(segment)
        arrays = committed_payload.arrays()
        n, m_count = arrays["templates"].shape
        try:
            solver._validate_energy("band_energy", b_new, m_count)
            solver._validate_error("band_error", e_new, m_count)
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc

        candidate = committed_payload.with_segment_readings(segment, energy, error)
        candidate_arrays = candidate.arrays()
        flat: solver.Confirmations = {
            (seg, key): pressed
            for seg, keys in confirmations_snapshot.items()
            for key, pressed in keys.items()
        }
        try:
            verdict = solver.classify(
                candidate_arrays["templates"],
                (candidate_arrays["b1"], candidate_arrays["b2"]),
                (candidate_arrays["e1"], candidate_arrays["e2"]),
                candidate_arrays["lo"],
                candidate_arrays["hi"],
                confirmations=flat,
                time_limit=candidate.time_limit,
            )
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc
        except solver.SolverInconclusive as exc:
            # The computation did not finish: keep the current readings and
            # report exactly as they were.
            raise ApiError("solver_inconclusive", str(exc), status=503) from exc

        if verdict.kind != "impossible":
            report = self._version_from_verdict(
                candidate, verdict, confirmations_snapshot, observed_version + 1
            )
            with self._lock:
                if analysis.current.version != observed_version:
                    raise ApiError(
                        "stale_version",
                        f"report advanced to version {analysis.current.version} during "
                        "recomputation; resubmit the readings against the new version",
                        status=409,
                        detail={
                            "current_version": analysis.current.version,
                            "expected_version": expected_version,
                        },
                    )
                analysis.payload = candidate
                analysis.reports.append(report)
                return analysis

        # Conflicting readings: compute the fewest bands to retest.  No band
        # subset is enumerated; one MILP minimises the retest count.
        try:
            plan = solver.minimum_retest_plan(
                arrays["templates"],
                b_old,
                e_old,
                b_new,
                e_new,
                arrays["lo"],
                arrays["hi"],
                segment=segment,
                confirmations=flat,
                time_limit=committed_payload.time_limit,
            )
        except solver.SolverError as exc:
            raise ApiError("invalid_input", str(exc)) from exc
        except solver.SolverInconclusive as exc:
            raise ApiError("solver_inconclusive", str(exc), status=503) from exc

        hybrid = committed_payload.with_segment_readings(
            segment,
            [float(v) for v in plan.effective_energy],
            [float(v) for v in plan.effective_error],
        )
        detail = {
            "current_version": observed_version,
            "expected_version": expected_version,
            "segment": segment,
            "minimum_retest_count": len(plan.retest_bands),
            "retest_bands": [committed_payload.band_names[j] for j in plan.retest_bands],
            "retest_band_indexes": plan.retest_bands,
            "witness": {
                "segment": segment,
                "pressed_keys": [
                    committed_payload.key_names[k] for k in plan.solution.pressed
                ],
                "gains": {
                    committed_payload.key_names[k]: float(plan.solution.gains[k])
                    for k in plan.solution.pressed
                },
            },
            # Readings the witness actually uses: listed bands reverted to the
            # current values, every other band keeping the submitted new ones.
            "hybrid_readings": hybrid.readings_view(),
            "submitted_readings": {
                "segment": segment,
                "energy": [float(v) for v in b_new],
                "error": [float(v) for v in e_new],
            },
        }
        raise ApiError(
            "retest_required",
            "the new readings conflict with a feasible reconstruction; retest the "
            "listed bands (their current values are restored for those bands only, "
            "submitted values are kept for every other band) and resubmit",
            status=409,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
def serialize_version(analysis: Analysis, report: ReportVersion) -> dict[str, Any]:
    payload = analysis.payload

    def confirmations(contrib: dict[int, dict[int, bool]]) -> dict[str, dict[str, bool]]:
        return {
            str(seg): {payload.key_names[k]: pressed for k, pressed in sorted(keys.items())}
            for seg, keys in sorted(contrib.items())
        }

    return {
        "analysis_id": analysis.analysis_id,
        "version": report.version,
        "kind": report.kind,
        "verdict": _verdict_text(report.kind),
        "created_at": report.created_at,
        "key_names": payload.key_names,
        "band_names": payload.band_names,
        "readings": report.readings,
        "possibilities": [
            {
                "hypothesis": name,
                "takes": [
                    {"segment": idx, "pressed_keys": take.pressed_keys, "gains": take.gains}
                    for idx, take in enumerate(takes)
                ],
            }
            for name, takes in report.possibilities
        ],
        "evidence": report.evidence,
        "confirmations": confirmations(report.contributions),
    }


def _verdict_text(kind: str) -> str:
    return {
        "same": "The two takes certainly use the same key set.",
        "different": "The two takes certainly use different key sets.",
        "both": "Either case is possible: both a shared and a differing key set fit the evidence.",
        "impossible": "No reconstruction exists; the recorded evidence conflicts.",
    }.get(kind, kind)


def serialize_confirmations(
    analysis: Analysis,
    current_version: int,
    confirmations: dict[int, dict[int, bool]],
) -> dict[str, Any]:
    payload = analysis.payload
    return {
        "analysis_id": analysis.analysis_id,
        "current_version": current_version,
        "confirmations": {
            str(seg): {payload.key_names[k]: pressed for k, pressed in sorted(keys.items())}
            for seg, keys in sorted(confirmations.items())
        },
    }
