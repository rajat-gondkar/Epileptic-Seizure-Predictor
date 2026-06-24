import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'
import { checkHealth, predictEdf, buildSimulatedResult, API_BASE } from '../lib/api'

const CLASS = [
  { name: 'Interictal', color: '#64748b', desc: 'Normal activity' },
  { name: 'Preictal', color: '#f59e0b', desc: 'Pre-seizure — alarm window' },
  { name: 'Ictal', color: '#ef4444', desc: 'Seizure activity' },
]

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
  return <canvas ref={ref} className="w-full" style={{ height: 90 }} />
}

function ProbBar({ label, value, color }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
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
  const [status, setStatus] = useState('checking') // checking | online | offline
  const [result, setResult] = useState(null)
  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const fileRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    checkHealth().then((r) => setStatus(r.online ? 'online' : 'offline'))
  }, [])

  // playback loop
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
    }, 350)
    return () => clearInterval(timerRef.current)
  }, [playing, result])

  const loadResult = useCallback((res) => {
    setResult(res)
    setPlayhead(0)
    setPlaying(true)
    setError(null)
  }, [])

  const onUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const res = await predictEdf(file)
      loadResult(res)
    } catch (err) {
      setError(err.message || 'Inference failed.')
    } finally {
      setBusy(false)
    }
  }

  const runSimulated = () => loadResult(buildSimulatedResult())

  const current = result?.windows?.[playhead]
  const currentClass = current ? CLASS[current.pred_class] : null

  return (
    <section id="demo" className="section-pad">
      <SectionHeader
        eyebrow="07 · Live Inference"
        title="Run the model on an EEG recording"
        subtitle="Upload a CHB-MIT .edf file and the trained BiLSTM classifies every 30-second window in real time — interictal, preictal, or ictal."
        accentDot="#22d3ee"
      />

      <Reveal>
        <div className="mt-10 glass p-6 sm:p-8 shadow-glow">
          {/* status + controls */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span
                className="inline-flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg border"
                style={{
                  color: status === 'online' ? '#34d399' : status === 'offline' ? '#fb923c' : '#94a3b8',
                  borderColor: status === 'online' ? '#34d39955' : status === 'offline' ? '#fb923c55' : '#ffffff22',
                  background: status === 'online' ? '#34d39912' : 'transparent',
                }}
              >
                <span className={`h-2 w-2 rounded-full ${status === 'checking' ? 'animate-pulse' : ''}`}
                  style={{ background: status === 'online' ? '#34d399' : status === 'offline' ? '#fb923c' : '#94a3b8' }} />
                {status === 'online' ? 'Model backend online' : status === 'offline' ? 'Backend offline' : 'Checking backend…'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <input ref={fileRef} type="file" accept=".edf" onChange={onUpload} className="hidden" />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={status !== 'online' || busy}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-eeg/15 text-eeg border border-eeg/30 hover:bg-eeg/25 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {busy ? 'Analyzing…' : 'Upload .edf & analyze'}
              </button>
              <button
                onClick={runSimulated}
                className="px-4 py-2 rounded-lg text-sm font-medium glass glass-hover text-slate-200"
              >
                Play sample (simulated)
              </button>
            </div>
          </div>

          {status === 'offline' && (
            <div className="mt-4 text-xs text-slate-500">
              Start the inference server to enable real uploads:{' '}
              <code className="text-slate-300">./venv/bin/python -m uvicorn src.api.main:app --port 8000</code>.
              Until then, the simulated playback demonstrates the workflow.
            </div>
          )}
          {error && <div className="mt-4 text-sm text-red-400">⚠ {error}</div>}

          {/* results */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6"
              >
                {result.simulated && (
                  <div className="mb-4 text-xs px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 inline-block">
                    Real-time Demo for Pre-Ictal Prediction
                  </div>
                )}

                <div className="grid lg:grid-cols-3 gap-4">
                  {/* live signal + current prediction */}
                  <div className="lg:col-span-2 rounded-xl bg-ink-950/60 border border-white/[0.05] p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-sm text-slate-400">
                        Window {playhead + 1}/{result.n_windows} ·{' '}
                        <span className="stat-num text-slate-300">
                          {current?.start_sec}s–{current?.end_sec}s
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setPlaying((p) => !p)}
                          className="text-xs px-3 py-1 rounded-md bg-white/[0.06] hover:bg-white/[0.12] text-slate-200"
                        >
                          {playing ? 'Pause' : 'Play'}
                        </button>
                        <button
                          onClick={() => { setPlayhead(0); setPlaying(true) }}
                          className="text-xs px-3 py-1 rounded-md bg-white/[0.06] hover:bg-white/[0.12] text-slate-200"
                        >
                          Restart
                        </button>
                      </div>
                    </div>

                    <WindowTrace preview={current?.preview} color={currentClass?.color} />

                    {/* timeline ribbon */}
                    <div className="mt-3">
                      <div className="flex h-7 rounded-md overflow-hidden border border-white/[0.06]">
                        {result.windows.map((w, i) => (
                          <button
                            key={i}
                            onClick={() => { setPlayhead(i); setPlaying(false) }}
                            title={`${w.start_sec}s · ${CLASS[w.pred_class].name}`}
                            className="flex-1 transition-all"
                            style={{
                              background: CLASS[w.pred_class].color,
                              opacity: i === playhead ? 1 : 0.5,
                              outline: i === playhead ? '2px solid #fff' : 'none',
                              outlineOffset: '-2px',
                            }}
                          />
                        ))}
                      </div>
                      <div className="mt-1 flex justify-between text-[10px] text-slate-600">
                        <span>0s</span>
                        <span>seizure timeline · click to scrub</span>
                        <span>{result.duration_sec}s</span>
                      </div>
                    </div>
                  </div>

                  {/* current probabilities */}
                  <div className="rounded-xl bg-ink-950/60 border border-white/[0.05] p-4 flex flex-col">
                    <div className="text-sm text-slate-400 mb-3">Model output</div>
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={current?.pred_class}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="text-center mb-4 py-3 rounded-lg"
                        style={{ background: `${currentClass?.color}1a`, border: `1px solid ${currentClass?.color}55` }}
                      >
                        <div className="text-lg font-semibold" style={{ color: currentClass?.color }}>
                          {currentClass?.name}
                        </div>
                        <div className="text-xs text-slate-500">{currentClass?.desc}</div>
                      </motion.div>
                    </AnimatePresence>
                    <div className="space-y-3">
                      <ProbBar label="Interictal" value={current?.p_interictal || 0} color="#64748b" />
                      <ProbBar label="Preictal" value={current?.p_preictal || 0} color="#f59e0b" />
                      <ProbBar label="Ictal" value={current?.p_ictal || 0} color="#ef4444" />
                    </div>
                  </div>
                </div>

                {/* summary */}
                <div className="mt-4 grid sm:grid-cols-4 gap-3">
                  <SummaryCard label="Recording" value={result.filename} small />
                  <SummaryCard label="Windows analyzed" value={`${result.n_windows} × 30s`} />
                  <SummaryCard label="Max preictal prob" value={`${(result.summary.max_preictal_prob * 100).toFixed(0)}%`} color="#f59e0b" />
                  <SummaryCard label="Max ictal prob" value={`${(result.summary.max_ictal_prob * 100).toFixed(0)}%`} color="#ef4444" />
                </div>

                <div className="mt-3 rounded-xl p-4 text-center font-medium"
                  style={{
                    background: result.summary.any_ictal ? '#ef444415' : result.summary.any_preictal ? '#f59e0b15' : '#34d39915',
                    border: `1px solid ${result.summary.any_ictal ? '#ef444455' : result.summary.any_preictal ? '#f59e0b55' : '#34d39955'}`,
                    color: result.summary.any_ictal ? '#fca5a5' : result.summary.any_preictal ? '#fcd34d' : '#6ee7b7',
                  }}>
                  {result.summary.any_ictal
                    ? '⚠ Seizure (ictal) activity detected in this recording'
                    : result.summary.any_preictal
                    ? '⚠ Pre-seizure (preictal) activity detected — would trigger an early-warning alert'
                    : '✓ No seizure-related activity detected'}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {!result && (
            <div className="mt-8 text-center text-sm text-slate-500 py-8">
              Upload an EEG recording or play the sample to see per-window seizure classification.
            </div>
          )}
        </div>
      </Reveal>

      <Reveal>
        <p className="mt-4 text-xs text-slate-600 text-center">
          Real inference uses the all-patients BiLSTM ({API_BASE}). CHB-MIT .edf files can be downloaded from{' '}
          <a className="text-eeg/70 hover:text-eeg" href="https://physionet.org/content/chbmit/1.0.0/" target="_blank" rel="noreferrer">
            physionet.org/content/chbmit
          </a>.
        </p>
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
