"""End-to-end tests for the re-extracted readings endpoint.

Covers: successful reading replacement with versioned retrieval, minimum
retest conflicts with witnesses, hybrid resubmission semantics, optimistic
concurrency (stale version and a version change during recomputation),
timeouts, and input validation.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app import store as store_mod
from app.main import STORE, app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def base_payload() -> dict:
    # Three orthogonal templates: keys C4, E4, G4 over bands f0, f1, f2.
    # Both takes currently reconstruct as C4 played at gain ~1.
    return {
        "key_names": ["C4", "E4", "G4"],
        "band_names": ["f0", "f1", "f2"],
        "templates": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "band_energy_1": [1.0, 0.0, 0.0],
        "band_error_1": [0.05, 0.05, 0.05],
        "band_energy_2": [1.0, 0.0, 0.0],
        "band_error_2": [0.05, 0.05, 0.05],
        "gain_lower": [0.5, 0.5, 0.5],
        "gain_upper": [2.0, 2.0, 2.0],
        "time_limit": 10.0,
    }


def create(client: TestClient, payload: dict | None = None) -> dict:
    resp = client.post("/analyses", json=payload or base_payload())
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_readings_endpoint_present_in_openapi() -> None:
    schema = app.openapi()
    assert "/analyses/{analysis_id}/readings" in schema["paths"]


def test_successful_reading_replacement_advances_version(client: TestClient) -> None:
    report = create(client)
    aid = report["analysis_id"]
    # Re-extracted segment 0 readings: slightly shifted but still reconstructable.
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.1, 0.02, 0.0],
            "band_error": [0.08, 0.08, 0.08],
        },
    )
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["version"] == 2
    assert updated["kind"] in {"same", "both"}
    # The new version carries the new segment-0 readings and unchanged segment 1.
    assert updated["readings"]["0"]["energy"] == [1.1, 0.02, 0.0]
    assert updated["readings"]["0"]["error"] == [0.08, 0.08, 0.08]
    assert updated["readings"]["1"]["energy"] == [1.0, 0.0, 0.0]


def test_versioned_retrieval_returns_readings_at_that_version(client: TestClient) -> None:
    report = create(client)
    aid = report["analysis_id"]
    client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 1,
            "band_energy": [0.95, 0.0, 0.01],
            "band_error": [0.1, 0.1, 0.1],
        },
    )
    v1 = client.get(f"/analyses/{aid}", params={"version": 1}).json()
    v2 = client.get(f"/analyses/{aid}", params={"version": 2}).json()
    assert v1["readings"]["1"]["energy"] == [1.0, 0.0, 0.0]
    assert v2["readings"]["1"]["energy"] == [0.95, 0.0, 0.01]
    current = client.get(f"/analyses/{aid}").json()
    assert current["version"] == 2
    assert current["readings"] == v2["readings"]


def test_conflicting_readings_return_minimum_retest_plan(client: TestClient) -> None:
    report = create(client)
    aid = report["analysis_id"]
    # Band f2 demands energy 5, unreachable with max gain 2 -> must be reverted.
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0, 0.0, 5.0],
            "band_error": [0.05, 0.05, 0.05],
        },
    )
    assert resp.status_code == 409, resp.text
    err = resp.json()["error"]
    assert err["code"] == "retest_required"
    detail = err["detail"]
    assert detail["minimum_retest_count"] == 1
    assert detail["retest_bands"] == ["f2"]
    assert detail["retest_band_indexes"] == [2]
    # Witness for the conflicting segment: its keys and gains.
    assert detail["witness"]["segment"] == 0
    assert detail["witness"]["pressed_keys"] == ["C4"]
    assert 0.5 <= detail["witness"]["gains"]["C4"] <= 2.0
    # The hybrid readings revert only the listed band to current values.
    hybrid = detail["hybrid_readings"]["0"]
    assert hybrid["energy"] == [1.0, 0.0, 0.0]  # f2 back to current 0.0
    assert hybrid["error"] == [0.05, 0.05, 0.05]


def test_conflict_does_not_change_current_state(client: TestClient) -> None:
    report = create(client)
    aid = report["analysis_id"]
    conflict = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [5.0, 5.0, 5.0],
            "band_error": [0.01, 0.01, 0.01],
        },
    )
    assert conflict.status_code == 409
    detail = conflict.json()["error"]["detail"]
    # Every new band excludes the feasible output (energy 5, tight error),
    # so all three bands must be reverted.
    assert detail["minimum_retest_count"] == 3
    assert detail["retest_bands"] == ["f0", "f1", "f2"]
    # Nothing was committed: still version 1 with original readings.
    current = client.get(f"/analyses/{aid}").json()
    assert current["version"] == 1
    assert current["readings"]["0"]["energy"] == [1.0, 0.0, 0.0]
    versions = client.get(f"/analyses/{aid}/versions").json()
    assert [v["version"] for v in versions["versions"]] == [1]


def test_hybrid_resubmission_succeeds_and_keeps_new_values_outside_list(
    client: TestClient,
) -> None:
    report = create(client)
    aid = report["analysis_id"]
    new_energy = [1.2, 0.1, 5.0]
    new_error = [0.1, 0.1, 0.05]
    conflict = client.post(
        f"/analyses/{aid}/readings",
        json={"expected_version": 1, "segment": 0,
              "band_energy": new_energy, "band_error": new_error},
    )
    assert conflict.status_code == 409
    detail = conflict.json()["error"]["detail"]
    assert detail["retest_bands"] == ["f2"]
    # Resubmit exactly the hybrid readings: f2 reverted, others kept new.
    hybrid = detail["hybrid_readings"]["0"]
    ok = client.post(
        f"/analyses/{aid}/readings",
        json={"expected_version": 1, "segment": 0,
              "band_energy": hybrid["energy"], "band_error": hybrid["error"]},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["version"] == 2
    assert body["readings"]["0"]["energy"][0] == 1.2  # new value kept
    assert body["readings"]["0"]["energy"][1] == 0.1  # new value kept
    assert body["readings"]["0"]["energy"][2] == 0.0  # reverted to current


def test_confirmations_carry_over_to_recomputed_report(client: TestClient) -> None:
    # Ambiguous A/B templates: both keys share one band.
    payload = {
        "key_names": ["A", "B"],
        "band_names": ["f"],
        "templates": [[1.0], [1.0]],
        "band_energy_1": [1.0],
        "band_error_1": [0.0],
        "band_energy_2": [1.0],
        "band_error_2": [0.0],
        "gain_lower": [0.5, 0.5],
        "gain_upper": [1.0, 1.0],
    }
    report = create(client, payload)
    aid = report["analysis_id"]
    confirmed = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "B", "pressed": True}],
        },
    )
    assert confirmed.status_code == 200
    # Now replace segment 0 readings with a still-feasible re-extraction.
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 2,
            "segment": 0,
            "band_energy": [0.9],
            "band_error": [0.2],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version"] == 3
    # Confirmation survived and still constrains every witness.
    assert body["confirmations"]["0"] == {"B": True}
    for possibility in body["possibilities"]:
        assert "B" in possibility["takes"][0]["pressed_keys"]


def test_retest_witness_honors_existing_confirmation(client: TestClient) -> None:
    payload = {
        "key_names": ["A", "B"],
        "band_names": ["f"],
        "templates": [[1.0], [1.0]],
        "band_energy_1": [1.0],
        "band_error_1": [0.0],
        "band_energy_2": [1.0],
        "band_error_2": [0.0],
        "gain_lower": [0.5, 0.5],
        "gain_upper": [1.0, 1.0],
    }
    report = create(client, payload)
    aid = report["analysis_id"]
    client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "B", "pressed": True}],
        },
    )
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 2,
            "segment": 0,
            "band_energy": [5.0],
            "band_error": [0.0],
        },
    )
    assert resp.status_code == 409
    witness = resp.json()["error"]["detail"]["witness"]
    assert "B" in witness["pressed_keys"]


def test_stale_expected_version_rejected(client: TestClient) -> None:
    report = create(client)
    aid = report["analysis_id"]
    # Advance to version 2 with a confirmation.
    client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 0, "key": "C4", "pressed": True}],
        },
    )
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0, 0.0, 0.0],
            "band_error": [0.05, 0.05, 0.05],
        },
    )
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert err["code"] == "stale_version"
    assert err["detail"]["current_version"] == 2
    # State untouched.
    assert client.get(f"/analyses/{aid}").json()["version"] == 2


def test_version_change_during_recomputation_rejects(client: TestClient, monkeypatch) -> None:
    report = create(client)
    aid = report["analysis_id"]
    entered = threading.Event()
    release = threading.Event()
    real_classify = store_mod.solver.classify
    calls = {"n": 0}

    def classify_with_barrier(*args, **kwargs):
        verdict = real_classify(*args, **kwargs)
        # Block after the readings recomputation finishes but before its
        # commit, so a concurrent editor can advance the version in the
        # meantime.  The confirmation request uses its own classify call.
        calls["n"] += 1
        if calls["n"] == 1:
            entered.set()
            release.wait(timeout=10)
        return verdict

    monkeypatch.setattr(store_mod.solver, "classify", classify_with_barrier)
    outcome: dict = {}

    def submit_readings() -> None:
        outcome["resp"] = client.post(
            f"/analyses/{aid}/readings",
            json={
                "expected_version": 1,
                "segment": 0,
                "band_energy": [1.05, 0.0, 0.0],
                "band_error": [0.1, 0.1, 0.1],
            },
        )

    worker = threading.Thread(target=submit_readings)
    worker.start()
    assert entered.wait(timeout=5)
    # The readings request has finished recomputation and is waiting to commit.
    # Reports remain queryable; a confirmation now advances the version.
    assert client.get(f"/analyses/{aid}").json()["version"] == 1
    confirm = client.post(
        f"/analyses/{aid}/confirmations",
        json={
            "expected_version": 1,
            "confirmations": [{"segment": 1, "key": "C4", "pressed": True}],
        },
    )
    assert confirm.status_code == 200
    release.set()
    worker.join(timeout=15)
    resp = outcome["resp"]
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "stale_version"
    # The reading replacement was not committed; the confirmation version is current.
    current = client.get(f"/analyses/{aid}").json()
    assert current["version"] == 2
    assert current["readings"]["0"]["energy"] == [1.0, 0.0, 0.0]


def test_inconclusive_solver_keeps_current_readings(client: TestClient, monkeypatch) -> None:
    report = create(client)
    aid = report["analysis_id"]

    def boom(*args, **kwargs):
        raise store_mod.solver.SolverInconclusive("time limit reached")

    monkeypatch.setattr(store_mod.solver, "classify", boom)
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0, 0.0, 0.0],
            "band_error": [0.05, 0.05, 0.05],
        },
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "solver_inconclusive"
    current = client.get(f"/analyses/{aid}").json()
    assert current["version"] == 1


def test_inconclusive_retest_model_also_keeps_current_state(
    client: TestClient, monkeypatch
) -> None:
    report = create(client)
    aid = report["analysis_id"]
    real_classify = store_mod.solver.classify

    def only_classify(*args, **kwargs):
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(store_mod.solver, "classify", only_classify)

    def boom(*args, **kwargs):
        raise store_mod.solver.SolverInconclusive("retest time limit reached")

    monkeypatch.setattr(store_mod.solver, "minimum_retest_plan", boom)
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [5.0, 0.0, 0.0],
            "band_error": [0.01, 0.01, 0.01],
        },
    )
    assert resp.status_code == 503
    assert client.get(f"/analyses/{aid}").json()["version"] == 1


def test_rejected_when_current_report_has_no_reconstruction(client: TestClient) -> None:
    payload = base_payload()
    payload["band_energy_1"] = [5.0, 0.0, 0.0]
    payload["band_energy_2"] = [5.0, 0.0, 0.0]
    payload["band_error_1"] = [0.01, 0.01, 0.01]
    payload["band_error_2"] = [0.01, 0.01, 0.01]
    report = create(client, payload)
    assert report["kind"] == "impossible"
    resp = client.post(
        f"/analyses/{report['analysis_id']}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0, 0.0, 0.0],
            "band_error": [0.05, 0.05, 0.05],
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "no_feasible_reconstruction"
    # Readings were not applied.
    current = client.get(f"/analyses/{report['analysis_id']}").json()
    assert current["readings"]["0"]["energy"] == [5.0, 0.0, 0.0]


def test_wrong_band_count_rejected(client: TestClient) -> None:
    report = create(client)
    resp = client.post(
        f"/analyses/{report['analysis_id']}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0, 0.0],
            "band_error": [0.05, 0.05, 0.05],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_negative_error_rejected(client: TestClient) -> None:
    report = create(client)
    resp = client.post(
        f"/analyses/{report['analysis_id']}/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0, 0.0, 0.0],
            "band_error": [0.05, -0.01, 0.05],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_unknown_analysis_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/analyses/does-not-exist/readings",
        json={
            "expected_version": 1,
            "segment": 0,
            "band_energy": [1.0],
            "band_error": [0.1],
        },
    )
    assert resp.status_code == 404


def test_second_segment_replacement_leaves_first_intact(client: TestClient) -> None:
    report = create(client)
    aid = report["analysis_id"]
    resp = client.post(
        f"/analyses/{aid}/readings",
        json={
            "expected_version": 1,
            "segment": 1,
            "band_energy": [0.8, 0.0, 0.0],
            "band_error": [0.2, 0.2, 0.2],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["readings"]["0"] == report["readings"]["0"]
    assert body["readings"]["1"]["energy"] == [0.8, 0.0, 0.0]
