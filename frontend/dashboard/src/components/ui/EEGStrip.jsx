import { useEffect, useRef, useMemo } from 'react'

/**
 * Realistic multichannel clinical-style EEG strip chart rendered on canvas.
 *
 * The signal is built from a sum of band-limited sinusoids with a 1/f-like
 * amplitude envelope (more power in lower frequencies, like real EEG), plus:
 *   - raw mode: slow baseline drift, 60 Hz line-noise ripple, eye-blink/
 *     movement artifacts on frontal channels, and higher broadband noise.
 *   - filtered mode: clean 0.5–45 Hz rhythms only (drift, line noise and
 *     large artifacts removed), matching the bandpass + z-score pipeline.
 */

// Bipolar montage labels (subset of the 19 common channels used by the model)
const CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'F3-C3', 'C3-P3', 'FP2-F4', 'F4-C4', 'CZ-PZ']

// Deterministic PRNG so the trace is stable across re-renders
function mulberry32(seed) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function buildChannelModel(seed) {
  const rnd = mulberry32(seed)
  // EEG bands with 1/f-style amplitude weighting
  const bands = [
    { lo: 1, hi: 4, amp: 1.0 }, // delta
    { lo: 4, hi: 8, amp: 0.7 }, // theta
    { lo: 8, hi: 13, amp: 0.85 }, // alpha (prominent)
    { lo: 13, hi: 30, amp: 0.35 }, // beta
    { lo: 30, hi: 45, amp: 0.15 }, // gamma
  ]
  const comps = []
  bands.forEach((b) => {
    const n = 3
    for (let i = 0; i < n; i++) {
      const f = b.lo + rnd() * (b.hi - b.lo)
      comps.push({ f, amp: b.amp * (0.5 + rnd()), phase: rnd() * Math.PI * 2 })
    }
  })
  // sparse irregular spike/transient times (seconds)
  const spikes = []
  let t = rnd() * 3
  while (t < 60) {
    spikes.push({ t, w: 0.04 + rnd() * 0.05, a: (rnd() > 0.5 ? 1 : -1) * (0.6 + rnd() * 0.8) })
    t += 1.5 + rnd() * 4
  }
  // eye-blink artifacts (slow, large) for frontal channels
  const blinks = []
  let bt = rnd() * 4
  while (bt < 60) {
    blinks.push({ t: bt, w: 0.18 + rnd() * 0.12, a: 1.4 + rnd() * 1.2 })
    bt += 3 + rnd() * 5
  }
  return { comps, spikes, blinks, driftPhase: rnd() * Math.PI * 2, driftF: 0.1 + rnd() * 0.2 }
}

function sampleSignal(model, t, filtered, frontal) {
  let y = 0
  // band-limited rhythms
  for (const c of model.comps) {
    y += c.amp * Math.sin(2 * Math.PI * c.f * t + c.phase)
  }
  y *= 6 // base µV-ish scale

  // sharp transients (kept in both, slightly damped when filtered)
  for (const s of model.spikes) {
    const d = t - s.t
    if (Math.abs(d) < s.w * 3) {
      const g = Math.exp(-(d * d) / (2 * s.w * s.w))
      y += s.a * g * (filtered ? 9 : 12)
    }
  }

  if (!filtered) {
    // slow baseline drift (removed by 0.5 Hz highpass)
    y += 22 * Math.sin(2 * Math.PI * model.driftF * t + model.driftPhase)
    // 60 Hz line-noise ripple (rolled off by the filter)
    y += 3.5 * Math.sin(2 * Math.PI * 60 * t)
    // broadband noise
    y += (Math.sin(t * 997.3) + Math.sin(t * 521.7) + Math.sin(t * 313.1)) * 1.6
    // eye-blink / movement artifacts, mostly on frontal channels
    if (frontal) {
      for (const b of model.blinks) {
        const d = t - b.t
        if (Math.abs(d) < b.w * 3) {
          const g = Math.exp(-(d * d) / (2 * b.w * b.w))
          y += b.a * g * 26
        }
      }
    }
  } else {
    // tiny residual noise only
    y += Math.sin(t * 211.3) * 0.6
  }
  return y
}

export default function EEGStrip({ filtered = true, height = 280, seizure = false }) {
  const canvasRef = useRef(null)
  const rafRef = useRef()
  const models = useMemo(() => CHANNELS.map((_, i) => buildChannelModel(1000 + i * 97)), [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    let width = canvas.offsetWidth

    const resize = () => {
      width = canvas.offsetWidth
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
    }
    resize()
    window.addEventListener('resize', resize)

    const labelW = 64
    const plotW = () => width - labelW
    const windowSec = 6 // seconds visible across the strip
    const laneH = height / CHANNELS.length
    const color = filtered ? '#34d399' : '#22d3ee'
    let globalT = 0

    const drawGrid = () => {
      ctx.clearRect(0, 0, width, height)
      // faint vertical second markers
      ctx.strokeStyle = 'rgba(255,255,255,0.04)'
      ctx.lineWidth = 1
      const pxPerSec = plotW() / windowSec
      const offset = (globalT * pxPerSec) % pxPerSec
      for (let x = labelW + plotW() - offset; x > labelW; x -= pxPerSec) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }
      // lane separators + labels
      ctx.fillStyle = 'rgba(148,163,184,0.7)'
      ctx.font = '10px JetBrains Mono, monospace'
      for (let i = 0; i < CHANNELS.length; i++) {
        const y = i * laneH
        ctx.strokeStyle = 'rgba(255,255,255,0.03)'
        ctx.beginPath()
        ctx.moveTo(labelW, y)
        ctx.lineTo(width, y)
        ctx.stroke()
        ctx.fillText(CHANNELS[i], 8, y + laneH / 2 + 3)
      }
    }

    const draw = () => {
      drawGrid()
      const pxPerSec = plotW() / windowSec
      const dt = windowSec / plotW()
      // seizure mode raises amplitude + rhythmic sharpness
      const seizGain = seizure ? 2.2 : 1
      for (let i = 0; i < CHANNELS.length; i++) {
        const model = models[i]
        const frontal = CHANNELS[i].startsWith('FP')
        const midY = i * laneH + laneH / 2
        ctx.beginPath()
        ctx.lineWidth = 1.1
        ctx.strokeStyle = seizure ? '#f87171' : color
        ctx.globalAlpha = 0.9
        for (let px = 0; px <= plotW(); px += 1) {
          const tAbs = globalT + (px / pxPerSec) - windowSec
          let v = sampleSignal(model, tAbs, filtered, frontal)
          if (seizure) {
            // rhythmic high-amplitude spike-wave during ictal
            v = v * seizGain + Math.sin(2 * Math.PI * 3.2 * tAbs) * 18 * Math.sin(2 * Math.PI * 0.5 * tAbs)
          }
          const y = midY - v * (laneH / 130)
          const x = labelW + px
          if (px === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
      ctx.globalAlpha = 1
      globalT += 0.016 // ~time advance per frame (slow, readable)
      rafRef.current = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [filtered, height, seizure, models])

  return <canvas ref={canvasRef} className="w-full block" style={{ height }} />
}
