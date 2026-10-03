from typing import Optional

from pydantic import BaseModel, model_validator


class AnalyzeRequest(BaseModel):
    tic_id: Optional[int] = None
    time: Optional[list[float]] = None
    flux: Optional[list[float]] = None
    flux_err: Optional[list[float]] = None
    include_plot: bool = False
    allow_download: bool = True

    @model_validator(mode="after")
    def _check(self):
        if self.tic_id is None and not (self.time and self.flux):
            raise ValueError("provide tic_id, or time and flux")
        if self.time and self.flux and len(self.time) != len(self.flux):
            raise ValueError("time and flux must have the same length")
        if self.flux_err and self.time and len(self.flux_err) != len(self.time):
            raise ValueError("flux_err must match time")
        return self


class BLSSignal(BaseModel):
    period_d: Optional[float]
    t0: Optional[float]
    duration_h: Optional[float]
    depth_ppm: Optional[float]
    snr: Optional[float]
    n_transits: int


class FitResult(BaseModel):
    ok: bool
    error: str = ""
    period_d: Optional[float] = None
    period_err_d: Optional[float] = None
    depth_ppm: Optional[float] = None
    depth_err_ppm: Optional[float] = None
    duration_h: Optional[float] = None
    duration_err_h: Optional[float] = None
    reduced_chi2: Optional[float] = None


class Classification(BaseModel):
    label: str
    score_planet: float
    score_planet_logreg: float
    note: str


class AnalyzeResponse(BaseModel):
    tic_id: Optional[int] = None
    detected: bool
    bls: BLSSignal
    fit: FitResult
    classification: Classification
    flags: list[str]
    plot_png_base64: Optional[str] = None
