export interface BLSSignal {
  period_d: number | null
  t0: number | null
  duration_h: number | null
  depth_ppm: number | null
  snr: number | null
  n_transits: number
}

export interface FitResult {
  ok: boolean
  error: string
  period_d: number | null
  period_err_d: number | null
  depth_ppm: number | null
  depth_err_ppm: number | null
  duration_h: number | null
  duration_err_h: number | null
  reduced_chi2: number | null
}

export interface Classification {
  label: string
  prob_planet: number
  prob_planet_logreg: number
  note: string
}

export interface AnalyzeResponse {
  tic_id: number | null
  detected: boolean
  bls: BLSSignal
  fit: FitResult
  classification: Classification
  flags: string[]
  plot_png_base64: string | null
}

export async function analyzeTic(ticId: number): Promise<AnalyzeResponse> {
  const r = await fetch('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tic_id: ticId, include_plot: true }),
  })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      if (typeof j.detail === 'string') msg = j.detail
    } catch {
      // keep the default message
    }
    throw new Error(msg)
  }
  return (await r.json()) as AnalyzeResponse
}
