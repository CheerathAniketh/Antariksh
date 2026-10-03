import { useEffect, useRef } from 'react'
import { animTime, fluxAt } from '../lib/orbit'
import type { OrbitParams } from '../lib/orbit'

export default function FluxTrace({ orbit }: { orbit: OrbitParams }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let raf = 0

    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const now = animTime()
      const span = orbit.periodS * 2.2
      const depth = orbit.rp * orbit.rp
      const lo = 1 - depth * 1.8
      const hi = 1 + depth * 0.35
      const yOf = (f: number) => h * (1 - (f - lo) / (hi - lo))

      ctx.strokeStyle = 'rgba(120,180,255,0.18)'
      ctx.setLineDash([4, 6])
      ctx.beginPath()
      ctx.moveTo(0, yOf(1))
      ctx.lineTo(w, yOf(1))
      ctx.stroke()
      ctx.setLineDash([])

      ctx.strokeStyle = '#5ee7ff'
      ctx.lineWidth = 1.8
      ctx.shadowColor = '#5ee7ff'
      ctx.shadowBlur = 8
      ctx.beginPath()
      for (let px = 0; px <= w; px += 2) {
        const t = now - span + (px / w) * span
        const y = yOf(fluxAt(t, orbit))
        if (px === 0) ctx.moveTo(px, y)
        else ctx.lineTo(px, y)
      }
      ctx.stroke()
      ctx.shadowBlur = 0

      raf = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(raf)
  }, [orbit])

  return <canvas ref={ref} className="h-24 w-full" />
}
