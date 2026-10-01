import base64

import lightkurve as lk
import numpy as np
from fastapi import FastAPI, HTTPException

from antariksh.api.schemas import AnalyzeRequest, AnalyzeResponse
from antariksh.config import RAW_DIR
from antariksh.pipeline import analyze_lightcurve
from antariksh.preprocess.pipeline import load_raw
from antariksh.viz.plots import fig_to_png, make_figure

app = FastAPI(title="Antariksh", version="0.1.0",
              description="TESS light curve -> BLS detection -> classification -> transit fit")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if req.tic_id is not None and not req.time:
        if not (RAW_DIR / f"TIC{req.tic_id}.fits").exists():
            if not req.allow_download:
                raise HTTPException(404, f"TIC{req.tic_id} is not cached and allow_download is false")
            from antariksh.ingest.mast import download_target
            row = download_target(req.tic_id)
            if row["status"] not in ("ok", "cached"):
                raise HTTPException(404, f"no SPOC 2-min light curve for TIC{req.tic_id} ({row['status']})")
        lc = load_raw(req.tic_id)
    else:
        t = np.asarray(req.time, float)
        f = np.asarray(req.flux, float)
        e = np.asarray(req.flux_err, float) if req.flux_err else np.full_like(f, np.nan)
        lc = lk.LightCurve(time=t, flux=f, flux_err=e)

    try:
        res = analyze_lightcurve(lc)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(422, f"analysis failed: {type(ex).__name__}: {ex}") from ex

    png = None
    if req.include_plot:
        title = f"TIC{req.tic_id}" if req.tic_id is not None else "uploaded light curve"
        png = base64.b64encode(fig_to_png(make_figure(res["_curve"], res["_bls"], res["_fit"], title))).decode()

    out = {k: v for k, v in res.items() if not k.startswith("_")}
    return AnalyzeResponse(tic_id=req.tic_id, plot_png_base64=png, **out)
