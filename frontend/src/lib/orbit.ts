export interface OrbitParams {
  periodS: number // animation seconds per orbit
  a: number // orbit radius in stellar radii (scene units)
  rp: number // planet/star radius ratio
}

export const DEMO_ORBIT: OrbitParams = { periodS: 4, a: 7, rp: 0.08 }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

const reduceMotion =
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

export function animTime(): number {
  return reduceMotion ? 1.2 : performance.now() / 1000
}

// circular orbit, central transit: T ~ P / (pi * a/Rs), depth ~ (Rp/Rs)^2
export function orbitFromSignal(periodD: number, durationH: number, depthPpm: number): OrbitParams {
  const t = Math.max(durationH, 0.05) / 24
  return {
    periodS: clamp(periodD * 0.8, 2.5, 12),
    a: clamp(periodD / (Math.PI * t), 3, 14),
    rp: clamp(Math.sqrt(Math.max(depthPpm, 1) * 1e-6), 0.01, 0.3),
  }
}

export function planetAngle(t: number, o: OrbitParams): number {
  return (2 * Math.PI * t) / o.periodS
}

// relative flux of the star at time t (1 = no transit)
export function fluxAt(t: number, o: OrbitParams): number {
  const th = planetAngle(t, o)
  if (Math.cos(th) <= 0) return 1
  const sep = Math.abs(o.a * Math.sin(th))
  const x = clamp((1 + o.rp - sep) / (2 * o.rp), 0, 1)
  return 1 - o.rp * o.rp * (x * x * (3 - 2 * x))
}
