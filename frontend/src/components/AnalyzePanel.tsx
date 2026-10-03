import { useState } from 'react'
import { analyzeTic } from '../api'
import type { AnalyzeResponse } from '../api'

const EXAMPLES = [
  { tic: 393831507, label: 'Recovered planet' },
  { tic: 159951311, label: 'Clean planet, scored low' },
  { tic: 437011608, label: 'Variability beats the planet' },
  { tic: 404421005, label: 'Systematic ramp' },
]

export default function AnalyzePanel({ onResult }: { onResult: (r: AnalyzeResponse) => void }) {
  const [tic, setTic] = useState('393831507')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run(value: string) {
    const id = Number.parseInt(value, 10)
    if (!Number.isFinite(id) || id <= 0) {
      setError('Enter a numeric TIC ID.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      onResult(await analyzeTic(id))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="glass p-6">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-dim">Target</p>
      <div className="mt-3 flex flex-col gap-3 sm:flex-row">
        <input
          value={tic}
          onChange={(e) => setTic(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && run(tic)}
          inputMode="numeric"
          placeholder="TIC ID"
          className="flex-1 rounded-lg border border-white/10 bg-black/40 px-4 py-3 font-mono text-lg text-ink outline-none focus:border-cyan-glow"
        />
        <button
          onClick={() => run(tic)}
          disabled={busy}
          className="rounded-lg bg-cyan-glow px-6 py-3 font-semibold text-void transition hover:brightness-110 disabled:opacity-50"
        >
          {busy ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.tic}
            disabled={busy}
            onClick={() => {
              setTic(String(ex.tic))
              run(String(ex.tic))
            }}
            className="rounded-full border border-white/10 px-3 py-1 text-xs text-dim transition hover:border-cyan-glow hover:text-ink disabled:opacity-50"
          >
            <span className="font-mono">{ex.tic}</span> · {ex.label}
          </button>
        ))}
      </div>

      {busy && (
        <p className="mt-4 animate-pulse font-mono text-sm text-cyan-glow">
          Running BLS search and transit fit. This takes about 10 to 15 seconds.
        </p>
      )}
      {error && <p className="mt-4 font-mono text-sm text-amber-flag">{error}</p>}
    </div>
  )
}
