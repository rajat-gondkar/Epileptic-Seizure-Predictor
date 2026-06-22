import { useEffect, useRef } from 'react'

/**
 * Animated EEG-style trace rendered on a canvas.
 * `filtered` toggles between a noisy raw-looking signal and a clean filtered one.
 */
export default function EEGWave({ color = '#22d3ee', filtered = true, height = 120, speed = 1, lines = 1 }) {
  const canvasRef = useRef(null)
  const rafRef = useRef()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let width = canvas.offsetWidth
    const dpr = window.devicePixelRatio || 1

    const resize = () => {
      width = canvas.offsetWidth
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.scale(dpr, dpr)
    }
    resize()
    window.addEventListener('resize', resize)

    let phase = 0
    const seeds = Array.from({ length: lines }, (_, i) => i * 13.7)

    const draw = () => {
      ctx.clearRect(0, 0, width, height)
      const laneH = height / lines
      seeds.forEach((seed, li) => {
        const midY = laneH * li + laneH / 2
        ctx.beginPath()
        ctx.lineWidth = 1.6
        ctx.strokeStyle = color
        ctx.globalAlpha = 0.85 - li * (0.4 / Math.max(lines, 1))
        for (let x = 0; x <= width; x += 2) {
          const t = x / width
          let y = 0
          // base rhythm
          y += Math.sin((t * 18 + phase) + seed) * 10
          y += Math.sin((t * 7 + phase * 0.6) + seed * 2) * 7
          if (!filtered) {
            // high-frequency noise + drift for the "raw" look
            y += Math.sin(t * 120 + phase * 4 + seed) * 5
            y += (Math.random() - 0.5) * 8
            y += Math.sin(t * 2 + phase * 0.2) * 9
          } else {
            y += Math.sin((t * 30 + phase * 1.3) + seed) * 3
          }
          const py = midY - y * (laneH / 90)
          if (x === 0) ctx.moveTo(x, py)
          else ctx.lineTo(x, py)
        }
        ctx.stroke()
      })
      ctx.globalAlpha = 1
      phase += 0.04 * speed
      rafRef.current = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [color, filtered, height, speed, lines])

  return <canvas ref={canvasRef} className="w-full" style={{ height }} />
}
