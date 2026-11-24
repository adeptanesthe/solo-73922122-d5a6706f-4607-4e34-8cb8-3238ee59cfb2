# Piano take key-set classifier

Decides whether two recorded attempts of a piano exercise use the **same set of
keys**, reports witnesses for every case the evidence allows, lets an editor
confirm or correct individual key presses against a **versioned** report, and
accepts **re-extracted spectral readings** for one segment with a
minimum-cardinality retest plan when the new measurements conflict.

Built with FastAPI + SciPy (HiGHS MILP). Start with Docker Compose.

## Run

```bash
docker compose up --build
# service: http://127.0.0.1:8000  (interactive docs at /docs)
python demo/run_demo.py                 # runs all demo beats against :8000
```

Local dev without Docker:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
python -m pytest -q                     # 57 tests
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/analyses` | Create an analysis; returns report version 1 |
| `GET` | `/analyses/{id}?version=N` | Fetch a report version (default: current), including the readings it was computed from |
| `GET` | `/analyses/{id}/versions` | List report versions |
| `GET` | `/analyses/{id}/confirmations` | Current editor confirmations |
| `POST` | `/analyses/{id}/confirmations` | Apply/confirm or correct edits against `expected_version` |
| `POST` | `/analyses/{id}/readings` | Replace one segment's re-extracted band readings against `expected_version` |

`POST /analyses` body: `key_names` (≤ 24), `band_names`, `templates` (keys ×
bands), `band_energy_1/2`, `band_error_1/2`, `gain_lower`, `gain_upper`,
optional `time_limit`. Gains are closed intervals and must be strictly positive
(a played key scales the whole template by a positive gain; a zero lower bound
would make "played at zero gain" indistinguishable from "not played").

Every report version carries a `readings` object with the per-segment
`energy`/`error` vectors used to compute it, so the readings of both segments
"at that time" are retrievable via `GET /analyses/{id}?version=N`.

Report body highlights:

- `kind`: `same` (certainly identical sets), `different` (certainly different),
  `both` (either case is possible), `impossible` (evidence conflicts).
- `possibilities[]`: one entry per feasible hypothesis, each holding both takes
  with their `pressed_keys` and per-key `gains`.
- `evidence`: solver feasibility per hypothesis; for `impossible`, both models'
  diagnostics (`reason: conflicting_evidence`).
- `confirmations`: the editor state this version was computed from.

## Model

Each key `k` has a non-negative template `T[k,:]`. For a take with gain vector
`g >= 0` (zero for unplayed keys, `lo[k] <= g[k] <= hi[k]` for played keys) the
bands are `g @ T` and must satisfy `|g @ T - b| <= e` for every band. The two
takes may use different gains.

Two MILPs are solved; neither enumerates key subsets:

- **same** — one shared binary vector `x`, per-take gains `g1,g2`, linked by
  `lo[k]·x[k] <= g[k] <= hi[k]·x[k]`.
- **different** — two binary vectors `x1,x2`, with `d[k] = |x1[k]-x2[k]|`
  linearised and `sum(d) >= 1`.

Outcome from feasibility:

| same | different | kind |
| --- | --- | --- |
| feasible | infeasible | `same` |
| infeasible | feasible | `different` |
| feasible | feasible | `both` |
| infeasible | infeasible | `impossible` |

Witnesses are extracted and re-verified numerically (closed-interval gains,
band residuals, unplayed gains exactly zero). An undecided model (time limit)
raises rather than guessing.

## Editing semantics

- The editor submits confirmations with the report `expected_version`.
- The service recomputes from the **full** confirmation set: previous
  confirmations merged with this request's edits (a repeated key with the same
  value is idempotent; a changed value is a correction).
- The new report version and the merged confirmations take effect **together**,
  atomically, only after recomputation succeeds.
- A stale `expected_version` is rejected (409 `stale_version`); the original
  confirmations and reports remain queryable.
- If the confirmed edits make the problem infeasible, the request is rejected
  (409 `infeasible_after_confirmations`) with evidence, and nothing is applied.
- Confirmations that force take 0 and take 1 to different states for a key are
  not an input error: they make the same-set hypothesis infeasible and let the
  different-set hypothesis answer (or produce `impossible` if nothing fits).

## Re-extracted readings (`POST /analyses/{id}/readings`)

After re-extracting the spectrum of one segment, the editor submits that
segment's new per-band energies and absolute errors together with the current
`expected_version`.

Request body: `expected_version`, `segment` (`0` or `1`), `band_energy[]`,
`band_error[]` (one entry per band).

Rules:

- The request is only accepted when the **current** report admits a feasible
  reconstruction (`kind != impossible`); otherwise it is rejected with
  `409 no_feasible_reconstruction`.
- All existing key confirmations carry over unchanged; the report is
  recomputed from the merged confirmation set against the new readings.
- When both segments can still be reconstructed, the new readings and the
  recomputed report take effect **together** as the next version. Every version
  stores its own reading snapshot, retrievable with
  `GET /analyses/{id}?version=N`.
- When the new readings make the evidence conflict, nothing is applied. The
  response is `409 retest_required` with a **minimum-cardinality** list of
  bands to re-measure and a witness for the segment:
  - `retest_bands` / `retest_band_indexes`: the fewest bands whose new
    readings must be reverted;
  - `witness`: the segment's pressed keys and per-key gains under the hybrid
    readings, honoring every editor confirmation for that segment;
  - `hybrid_readings`: listed bands reverted to the current values, every
    other band keeping the submitted new values. Resubmitting these readings
    (with the same `expected_version`) applies the change.
- The minimum is found by **one MILP** (`min Σ zⱼ` with piecewise-linear
  hybrid band intervals); band combinations are never enumerated.
- Reads and confirmation submissions remain available while the solver runs.
  If the report version advances before the result is committed, the request
  fails with `409 stale_version` and must be resubmitted against the new
  version.
- If a model cannot finish within the time limit the response is
  `503 solver_inconclusive` (the computation simply did not complete); on
  conflict or timeout the current readings and report stay in force.

### Example requests

Successful replacement (new readings and recomputed report take effect
together):

```bash
curl -sS -X POST http://127.0.0.1:8000/analyses/$ID/readings \
  -H 'Content-Type: application/json' \
  -d '{
        "expected_version": 1,
        "segment": 0,
        "band_energy": [1.08, 0.01, 0.46],
        "band_error":  [0.06, 0.04, 0.06]
      }'
# 200 -> version 2 whose "readings" holds the new segment-0 vectors and the
# unchanged segment-1 vectors
```

Conflicting re-extraction (requires retesting):

```bash
curl -sS -X POST http://127.0.0.1:8000/analyses/$ID/readings \
  -H 'Content-Type: application/json' \
  -d '{
        "expected_version": 1,
        "segment": 0,
        "band_energy": [1.02, 0.0, 4.5],
        "band_error":  [0.05, 0.03, 0.05]
      }'
# 409 retest_required -> detail.retest_bands names the fewest bands to
# re-measure (e.g. ["f_G4"]), detail.witness gives the segment's keys/gains,
# and detail.hybrid_readings can be resubmitted verbatim.
```

## Demo beats (`demo/`)

1. `01_same.json` → `same`, both takes use C4+G4.
2. `02_different.json` → `different` (C4+G4 vs E4+G4).
3. `03_impossible.json` → `impossible`, conflicting evidence.
4. `04_ambiguous.json` → `both`; apply `04_confirmation.json` to edit, then a
   replayed request with the same `expected_version` returns `stale_version`
   while version 1 stays retrievable.
5. `05_readings_replacement.json` → re-extracted segment-0 readings that still
   reconstruct: accepted as a new version; versioned readings are fetched back.
6. `06_readings_conflict.json` → conflicting re-extraction: `retest_required`
   with the minimum retest list and a witness; the returned hybrid readings are
   resubmitted and accepted.

## Implementation notes / limits

- State is in-memory (single process); restarting the service clears analyses.
  The store is lock-protected, and both solver flows run outside the lock with a
  version re-check before commit.
- Solver time limit defaults to 10 s per model (`time_limit` field, ≤ 120 s).
- Templates must be non-negative and finite; band errors non-negative.
- The retest MILP introduces one binary per band (`z[j]`) and linearises the
  hybrid intervals with big-M constants; the objective is the retest count, so
  the returned list is provably minimum without enumerating band subsets.
