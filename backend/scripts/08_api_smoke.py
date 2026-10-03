"""API smoke receipt: runs four checks through the FastAPI app and records what actually happened.

Checks: (1) a real cached/downloadable target, (2) an injected transit, (3) noise only, (4) empty request.
Writes artifacts/receipts/api_smoke.json with the git commit, per-check outcome and key values.
A check that errors is recorded as failed with the error text; the script never hides a failure.
Run from the repo root after the pipeline has produced artifacts/models/baseline.joblib.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fastapi.testclient import TestClient  # noqa: E402
from test_bls_synthetic import make_curve  # noqa: E402

from antariksh.api.main import app  # noqa: E402
from antariksh.config import ARTIFACTS_DIR, ROOT  # noqa: E402

REAL_TIC = 393831507
client = TestClient(app)


def post_curve(with_transit):
    t, f, e = make_curve(with_transit)
    return client.post("/analyze", json={"time": t.tolist(), "flux": f.tolist(), "flux_err": e.tolist()})


def check_real():
    r = client.post("/analyze", json={"tic_id": REAL_TIC})
    j = r.json()
    ok = r.status_code == 200 and j["bls"]["period_d"] is not None and j["fit"]["ok"]
    return ok, {"http": r.status_code, "detected": j.get("detected"), "bls_period_d": j.get("bls", {}).get("period_d"),
                "fit_ok": j.get("fit", {}).get("ok"), "score_planet": j.get("classification", {}).get("score_planet")}


def check_injected():
    r = post_curve(True)
    j = r.json()
    p = j["bls"]["period_d"]
    return (r.status_code == 200 and j["detected"] and abs(p - 3.7) / 3.7 < 0.01), \
        {"http": r.status_code, "detected": j["detected"], "bls_period_d": p, "injected_period_d": 3.7}


def check_noise():
    r = post_curve(False)
    j = r.json()
    return (r.status_code == 200 and not j["detected"]), \
        {"http": r.status_code, "detected": j["detected"], "bls_snr": j["bls"]["snr"]}


def check_empty():
    r = client.post("/analyze", json={})
    return r.status_code == 422, {"http": r.status_code}


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    results = {}
    for name, fn in [("real_target_TIC%d" % REAL_TIC, check_real), ("injected_transit", check_injected),
                     ("noise_only_not_detected", check_noise), ("empty_request_422", check_empty)]:
        try:
            ok, detail = fn()
        except Exception as exc:  # recorded, not hidden
            ok, detail = False, {"error": f"{type(exc).__name__}: {exc}"}
        results[name] = {"passed": bool(ok), **detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    out = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit,
           "working_tree_dirty": dirty, "all_passed": all(v["passed"] for v in results.values()), "checks": results}
    (ARTIFACTS_DIR / "receipts").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "receipts" / "api_smoke.json").write_text(json.dumps(out, indent=2))
    print("saved artifacts/receipts/api_smoke.json")


if __name__ == "__main__":
    main()
