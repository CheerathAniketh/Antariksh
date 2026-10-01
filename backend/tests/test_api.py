import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from test_bls_synthetic import make_curve

from antariksh.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_empty_request():
    assert client.post("/analyze", json={}).status_code == 422


def test_analyze_uploaded_synthetic_light_curve():
    t, f, e = make_curve(True)
    r = client.post("/analyze", json={"time": t.tolist(), "flux": f.tolist(), "flux_err": e.tolist(),
                                      "include_plot": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["detected"]
    assert abs(j["bls"]["period_d"] - 3.7) / 3.7 < 0.01
    assert j["fit"]["ok"] and j["fit"]["depth_err_ppm"] > 0
    assert 0.0 <= j["classification"]["prob_planet"] <= 1.0
    assert j["plot_png_base64"]
