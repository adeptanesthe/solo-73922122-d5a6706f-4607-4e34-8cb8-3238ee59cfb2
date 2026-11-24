#!/usr/bin/env python3
"""Run the demo beats against a live service.

Usage:
    python demo/run_demo.py [base_url]

Defaults to http://127.0.0.1:8000 (the docker compose port).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent


def call(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def show(label: str, status: int, body: dict) -> None:
    print(f"\n=== {label} -> HTTP {status}, kind={body.get('kind')}, version={body.get('version')}")
    print(body.get("verdict", ""))
    for possibility in body.get("possibilities", []):
        print(f"  hypothesis: {possibility['hypothesis']}")
        for take in possibility["takes"]:
            gains = {k: round(v, 4) for k, v in take["gains"].items()}
            print(f"    take {take['segment']}: keys={take['pressed_keys']} gains={gains}")


def main() -> None:
    print("health:", call("GET", "/healthz"))

    for name, label in [
        ("01_same.json", "same key set"),
        ("02_different.json", "different key sets"),
        ("03_impossible.json", "conflicting evidence"),
        ("04_ambiguous.json", "both cases possible"),
    ]:
        status, body = call("POST", "/analyses", load(name))
        show(label, status, body)

    # Editor flow on the ambiguous analysis: correct take 0's key B.
    status, body = call("POST", "/analyses", load("04_ambiguous.json"))
    analysis_id = body["analysis_id"]
    version = body["version"]
    status, updated = call(
        "POST", f"/analyses/{analysis_id}/confirmations", load("04_confirmation.json")
    )
    show("after editor confirmation", status, updated)
    print("  confirmations:", updated.get("confirmations"))

    status, rejected = call(
        "POST", f"/analyses/{analysis_id}/confirmations", load("04_confirmation.json")
    )
    print("\n=== replayed stale confirmation ->", status, rejected.get("error", {}).get("code"))

    status, original = call("GET", f"/analyses/{analysis_id}?version={version}")
    print(f"=== original version {version} still queryable -> HTTP {status}, kind={original.get('kind')}")

    # Re-extracted readings beat on a fresh "same key set" analysis.
    status, base = call("POST", "/analyses", load("01_same.json"))
    readings_id = base["analysis_id"]
    print(f"\n=== readings analysis created -> HTTP {status}, id={readings_id}, version=1")

    # 1) Replacement that still admits a reconstruction: new readings and the
    #    recomputed report take effect together as version 2.
    status, replaced = call(
        "POST", f"/analyses/{readings_id}/readings", load("05_readings_replacement.json")
    )
    show("readings replacement accepted", status, replaced)
    if status == 200:
        print("  version-2 readings:", replaced.get("readings"))

    # 2) Conflicting re-extraction: minimum retest list plus a witness.
    status, conflict = call(
        "POST", f"/analyses/{readings_id}/readings", load("06_readings_conflict.json")
    )
    print("\n=== conflicting readings -> HTTP", status, conflict.get("error", {}).get("code"))
    if status == 409:
        detail = conflict["error"]["detail"]
        print("  retest bands:", detail["retest_bands"],
              "(count =", detail["minimum_retest_count"], ")")
        print("  witness:", detail["witness"])
        # Resubmit the hybrid readings: listed bands revert to current values,
        # every other band keeps the submitted new values.
        hybrid = {
            "expected_version": 2,
            "segment": detail["segment"],
            "band_energy": detail["hybrid_readings"][str(detail["segment"])]["energy"],
            "band_error": detail["hybrid_readings"][str(detail["segment"])]["error"],
        }
        status, healed = call(
            "POST", f"/analyses/{readings_id}/readings", hybrid
        )
        show("hybrid resubmission accepted", status, healed)

    # Readings used by each version remain retrievable by version.
    for v in (1, 2, 3):
        status, row = call("GET", f"/analyses/{readings_id}?version={v}")
        if status == 200:
            print(f"=== version {v} stored readings: {row['readings']['0']}")


if __name__ == "__main__":
    main()
