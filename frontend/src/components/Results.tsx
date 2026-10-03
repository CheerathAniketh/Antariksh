import { motion } from 'framer-motion'
import type { AnalyzeResponse } from '../api'

const fmt = (v: number | null | undefined, d = 2) => (v == null ? 'n/a' : v.toFixed(d))

function Stat({ label, value, err, unit }: { label: string; value: string; err?: string; unit: string }) {
  return (
    <div className="glass p-4">
      <p className="font-mono text-xs uppercase tracking-[0.18em] text-dim">{label}</p>
      <p className="mt-2 font-mono text-2xl text-ink">
        {value}
        <span className="ml-1 text-sm text-dim">{unit}</span>
      </p>
      {err && <p className="mt-1 font-mono text-xs text-dim">± {err}</p>}
    </div>
  )
}

export default function Results({ res }: { res: AnalyzeResponse }) {
  const { bls, fit, classification: c } = res
  const pct = Math.round(c.prob_planet * 100)

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mt-8 space-y-6"
    >
      <div className="glass p-6">
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`rounded-full px-3 py-1 font-mono text-xs uppercase tracking-widest ${
              res.detected ? 'bg-cyan-glow/15 text-cyan-glow' : 'bg-white/5 text-dim'
            }`}
          >
            {res.detected ? 'Signal detected' : 'No significant signal'}
          </span>
          <span className="font-mono text-sm text-ink">{c.label}</span>
          <span className="font-mono text-sm text-dim">SNR {fmt(bls.snr, 1)}</span>
          <span className="font-mono text-sm text-dim">{bls.n_transits} transits</span>
        </div>

        <div className="mt-5">
          <div className="flex items-baseline justify-between font-mono text-xs text-dim">
            <span>P(planet), uncalibrated</span>
            <span className="text-ink">{pct}%</span>
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded bg-white/10">
            <div className="h-full bg-cyan-glow" style={{ width: `${pct}%` }} />
          </div>
          <p className="mt-2 font-mono text-xs text-dim">
            logistic regression: {Math.round(c.prob_planet_logreg * 100)}%
          </p>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {res.flags.length === 0 ? (
            <span className="font-mono text-xs text-dim">no vetting flags raised</span>
          ) : (
            res.flags.map((f) => (
              <span
                key={f}
                className="rounded-full border border-amber-flag/40 px-3 py-1 font-mono text-xs text-amber-flag"
              >
                {f}
              </span>
            ))
          )}
        </div>
        <p className="mt-4 text-xs text-dim">{c.note}</p>
      </div>

      {fit.ok ? (
        <div className="grid gap-4 sm:grid-cols-3">
          <Stat label="Orbital period" value={fmt(fit.period_d, 4)} err={fmt(fit.period_err_d, 4)} unit="d" />
          <Stat label="Transit depth" value={fmt(fit.depth_ppm, 0)} err={fmt(fit.depth_err_ppm, 0)} unit="ppm" />
          <Stat label="Duration (T14)" value={fmt(fit.duration_h, 2)} err={fmt(fit.duration_err_h, 2)} unit="h" />
        </div>
      ) : (
        <div className="glass p-4 font-mono text-sm text-amber-flag">
          Transit fit unavailable{fit.error ? `: ${fit.error}` : '.'}
        </div>
      )}

      {res.plot_png_base64 && (
        <div className="glass overflow-hidden p-2">
          <img
            src={`data:image/png;base64,${res.plot_png_base64}`}
            alt="Four-panel diagnostic: raw light curve, periodogram, folded transit and fit"
            className="w-full rounded-lg"
          />
        </div>
      )}
    </motion.section>
  )
}
