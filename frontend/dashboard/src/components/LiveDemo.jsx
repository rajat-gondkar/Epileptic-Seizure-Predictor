import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'
import RiskGauge from './RiskGauge'
import fusion from '../data/fusion.json'
import { alertForScore } from '../lib/format'
import { parseEgf } from '../lib/egf'

function WindowTrace({ preview, color }) {
  const ref = useRef(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.offsetWidth
    const h = canvas.offsetHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)
    if (!preview || !preview.length) return
    ctx.strokeStyle = color
    ctx.lineWidth = 1.6
    ctx.beginPath()
    preview.forEach((v, i) => {
      const x = (i / (preview.length - 1)) * w
      const y = h / 2 - v * (h / 2.6)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
  }, [preview, color])
  return <canvas ref={ref} className="w-full" style={{ height: 84 }} />
}

function ScoreBar({ label, value, color, sub }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}{sub && <span className="text-slate-600"> · {sub}</span>}</span>
        <span className="stat-num" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </div>
  )
}

export default function LiveDemo() {
  const [result, setResult] = useState(null)
  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const fileRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!playing || !result) return
    timerRef.current = setInterval(() => {
      setPlayhead((p) => {
        if (p >= result.n_windows - 1) {
          setPlaying(false)
          return p
        }
        return p + 1
      })
    }, 320)
    return () => clearInterval(timerRef.current)
  }, [playing, result])

  const loadSample = useCallback((res) => {
    setResult(res)
    setPlayhead(0)
    setPlaying(true)
    setError(null)
  }, [])

  const handleText = (text, name) => {
    try {
      const res = parseEgf(text)
      res.filename = name
      loadSample(res)
    } catch (err) {
      setError(err.message || 'Could not read the fusion bundle.')
      setResult(null)
    }
  }

  const onUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const text = await file.text()
      handleText(text, file.name)
    } catch {
      setError('Failed to read the file.')
    } finally {
      setBusy(false)
      e.target.value = ''
    }
  }

  const current = result?.windows?.[playhead]
  const alert = current ? alertForScore(current.fused, fusion.alert_levels) : null
  const peakAlert = result ? alertForScore(result.summary.max_fused, fusion.alert_levels) : null

  return (
    <section id="demo" className="section-pad">
      <SectionHeader
        eyebrow="07 · Live Fusion Inference"
        title="Run the fusion model on a patient bundle"
        subtitle="Upload an .egf bundle (EEG window scores + genetic profile) and watch the attention-gated fusion model produce a per-window risk score and clinical alert."
        accentDot="#34d399"
      />

      <Reveal>
        <div className="mt-10 glass p-6 sm:p-8 shadow-glow-emerald">
          {/* controls */}
          <div className="flex items-center justify-end gap-4">
            <input ref={fileRef} type="file" accept=".egf" onChange={onUpload} className="hidden" />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={busy}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-fuse/15 text-fuse border border-fuse/30 hover:bg-fuse/25 transition-colors disabled:opacity-40"
            >
              {busy ? 'Reading…' : 'Upload .egf bundle'}
            </button>
          </div>

          {error && <div className="mt-4 text-sm text-red-400">⚠ {error}</div>}

          <AnimatePresence>
            {result && (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mt-6">
                {/* patient header */}
                <div className="flex flex-wrap items-center gap-x-6 gap-y-1 mb-4 text-sm">
                  <span className="text-slate-300">Patient <span className="stat-num text-white">{result.patient}</span></span>
                  <span className="text-slate-500">·</span>
                  <span className="text-slate-400">Genetic risk <span className="stat-num text-gene">{(result.genetic_score * 100).toFixed(0)}%</span></span>
                  {result.genes.length > 0 && (
                    <span className="flex items-center gap-1.5">
                      {result.genes.map((g) => (
                        <span key={g} className="text-[11px] font-mono px-2 py-0.5 rounded bg-gene/15 text-gene">{g}</span>
                      ))}
                    </span>
                  )}
                  <span className="text-slate-500">·</span>
                  <span className="text-slate-400">α gate <span className="stat-num text-fuse">{result.alpha.toFixed(2)}</span></span>
                </div>

                <div className="grid lg:grid-cols-3 gap-4">
                  {/* signal + timeline */}
                  <div className="lg:col-span-2 rounded-xl bg-ink-950/60 border border-white/[0.05] p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-sm text-slate-400">
                        Window {playhead + 1}/{result.n_windows} ·{' '}
                        <span className="stat-num text-slate-300">{current?.start_sec}s–{current?.end_sec}s</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => setPlaying((p) => !p)} className="text-xs px-3 py-1 rounded-md bg-white/[0.06] hover:bg-white/[0.12] text-slate-200">
                          {playing ? 'Pause' : 'Play'}
                        </button>
                        <button onClick={() => { setPlayhead(0); setPlaying(true) }} className="text-xs px-3 py-1 rounded-md bg-white/[0.06] hover:bg-white/[0.12] text-slate-200">
                          Restart
                        </button>
                      </div>
                    </div>

                    <WindowTrace preview={current?.preview} color={alert?.color} />

                    {/* timeline ribbon coloured by fused alert level */}
                    <div className="mt-3">
                      <div className="flex h-7 rounded-md overflow-hidden border border-white/[0.06]">
                        {result.windows.map((w, i) => {
                          const a = alertForScore(w.fused, fusion.alert_levels)
                          return (
                            <button
                              key={i}
                              onClick={() => { setPlayhead(i); setPlaying(false) }}
                              title={`${w.start_sec}s · ${a.name} (${w.fused.toFixed(2)})`}
                              className="flex-1 transition-all"
                              style={{
                                background: a.color,
                                opacity: i === playhead ? 1 : 0.45,
                                outline: i === playhead ? '2px solid #fff' : 'none',
                                outlineOffset: '-2px',
                              }}
                            />
                          )
                        })}
                      </div>
                      <div className="mt-1 flex justify-between text-[10px] text-slate-600">
                        <span>0s</span>
                        <span>fused-risk alert timeline · click to scrub</span>
                        <span>{result.duration_sec}s</span>
                      </div>
                    </div>

                    {/* branch breakdown */}
                    <div className="mt-4 space-y-3">
                      <ScoreBar label="EEG branch" sub="BiLSTM preictal prob" value={current?.eeg_score || 0} color="#22d3ee" />
                      <ScoreBar label="Genetic branch" sub="XGBoost risk" value={current?.genetic_score || 0} color="#a78bfa" />
                      <div className="pt-1 flex items-center gap-3 text-xs text-slate-500">
                        <span>Fusion gate:</span>
                        <span className="flex-1 flex h-1.5 rounded-full overflow-hidden">
                          <span style={{ width: `${result.alpha * 100}%`, background: '#22d3ee' }} />
                          <span style={{ width: `${(1 - result.alpha) * 100}%`, background: '#a78bfa' }} />
                        </span>
                        <span className="stat-num">α={result.alpha.toFixed(2)}</span>
                      </div>
                      <p className="text-[11px] text-slate-600 leading-relaxed">
                        α is the EEG weight set by the attention gate. The genetic risk is fixed per patient
                        (genes don't change over time); stronger pathogenic variants shift trust toward the
                        genetic branch — a lower α and a higher constant risk floor.
                      </p>
                    </div>
                  </div>

                  {/* fused output gauge */}
                  <div className="rounded-xl bg-ink-950/60 border border-white/[0.05] p-4 flex flex-col items-center">
                    <div className="text-sm text-slate-400 self-start mb-1">Fused risk · P_final</div>
                    <RiskGauge value={current?.fused || 0} color={alert?.color || '#34d399'} levels={fusion.alert_levels} />
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={alert?.level}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="text-center mt-1"
                      >
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg font-semibold text-sm"
                          style={{ background: `${alert?.color}1f`, color: alert?.color, border: `1px solid ${alert?.color}55` }}>
                          <span className="h-2 w-2 rounded-full animate-pulse" style={{ background: alert?.color }} />
                          Level {alert?.level} · {alert?.name}
                        </div>
                        <p className="text-xs text-slate-500 mt-1.5">{alert?.desc}</p>
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>

                {/* summary */}
                <div className="mt-4 grid sm:grid-cols-4 gap-3">
                  <SummaryCard label="Bundle" value={result.filename} small />
                  <SummaryCard label="Windows" value={`${result.n_windows} × 30s`} />
                  <SummaryCard label="Peak EEG branch" value={`${(result.summary.max_eeg * 100).toFixed(0)}%`} color="#22d3ee" />
                  <SummaryCard label="Peak fused risk" value={`${(result.summary.max_fused * 100).toFixed(0)}%`} color={peakAlert?.color} />
                </div>

                <div className="mt-3 rounded-xl p-4 text-center font-medium"
                  style={{ background: `${peakAlert?.color}15`, border: `1px solid ${peakAlert?.color}55`, color: peakAlert?.color }}>
                  {peakAlert?.level >= 4 && '⚠ Critical seizure risk detected — immediate intervention indicated'}
                  {peakAlert?.level === 3 && '⚠ High seizure risk — preictal activity flagged, alert care team'}
                  {peakAlert?.level === 2 && '◆ Moderate seizure risk — increased observation advised'}
                  {peakAlert?.level === 1 && '✓ Low risk — no seizure-related activity detected'}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {!result && (
            <div className="mt-8 text-center text-sm text-slate-500 py-8">
              Upload an <code className="text-slate-400">.egf</code> bundle to run the fusion model.
            </div>
          )}
        </div>
      </Reveal>
    </section>
  )
}

function SummaryCard({ label, value, color = '#e2e8f0', small }) {
  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/[0.05] p-3">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className={`stat-num ${small ? 'text-xs break-all' : 'text-lg'}`} style={{ color }}>{value}</div>
    </div>
  )
}
