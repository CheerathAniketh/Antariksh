import { lazy, Suspense, useState } from 'react'
import AnalyzePanel from './components/AnalyzePanel'
import FluxTrace from './components/FluxTrace'
import Results from './components/Results'
import { DEMO_ORBIT, orbitFromSignal } from './lib/orbit'
import type { OrbitParams } from './lib/orbit'
import type { AnalyzeResponse } from './api'

const Hero3D = lazy(() => import('./components/Hero3D'))

function orbitFromResult(r: AnalyzeResponse): OrbitParams | null {
  const f = r.fit
  if (f.ok && f.period_d != null && f.duration_h != null && f.depth_ppm != null && f.depth_ppm > 0) {
    return orbitFromSignal(f.period_d, f.duration_h, f.depth_ppm)
  }
  const b = r.bls
  if (b.period_d != null && b.duration_h != null && b.depth_ppm != null && b.depth_ppm > 0) {
    return orbitFromSignal(b.period_d, b.duration_h, b.depth_ppm)
  }
  return null
}

export default function App() {
  const [orbit, setOrbit] = useState<OrbitParams>(DEMO_ORBIT)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)

  function handleResult(r: AnalyzeResponse) {
    setResult(r)
    const o = orbitFromResult(r)
    if (o) setOrbit(o)
  }

  const depthPpm = Math.round(orbit.rp * orbit.rp * 1e6)

  return (
    <>
      <section className="relative h-[92vh] min-h-[640px] overflow-hidden">
        <div className="absolute inset-0">
          <Suspense fallback={null}>
            <Hero3D orbit={orbit} />
          </Suspense>
        </div>
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-void/60 via-transparent to-void" />

        <div className="relative z-10 mx-auto flex h-full max-w-6xl flex-col justify-between px-6 py-10">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.35em] text-cyan-glow">TESS · transit photometry</p>
            <h1 className="mt-4 text-6xl font-bold tracking-[0.12em] text-ink sm:text-8xl">ANTARIKSH</h1>
            <p className="mt-5 max-w-xl text-lg text-dim">
              Find tiny periodic dips in noisy starlight, classify them, and measure period, depth and duration
              with uncertainties.
            </p>
            <a
              href="#analyze"
              className="pointer-events-auto mt-8 inline-block rounded-lg border border-cyan-glow/60 px-6 py-3 font-mono text-sm uppercase tracking-widest text-cyan-glow transition hover:bg-cyan-glow/10"
            >
              Analyze a star
            </a>
          </div>

          <div className="glass pointer-events-auto max-w-2xl p-4">
            <div className="flex items-center justify-between font-mono text-xs text-dim">
              <span>model light curve · {result ? `TIC ${result.tic_id}` : 'demo orbit'}</span>
              <span>depth {depthPpm} ppm</span>
            </div>
            <FluxTrace orbit={orbit} />
            <p className="mt-1 text-[11px] text-dim">
              {result
                ? 'Drawn from the fitted period, duration and depth (circular orbit, central transit). Planet size is exaggerated when too small to see.'
                : 'Run an analysis and the orbit and curve follow the fitted parameters.'}
            </p>
          </div>
        </div>
      </section>

      <main id="analyze" className="mx-auto max-w-4xl px-6 pb-24 pt-10">
        <AnalyzePanel onResult={handleResult} />
        {result && <Results res={result} />}
        <footer className="mt-16 border-t border-white/10 pt-6 font-mono text-xs text-dim">
          TESS SPOC 2-min light curves · BLS detection · batman transit fit · feature-based classifier, uncalibrated
        </footer>
      </main>
    </>
  )
}
