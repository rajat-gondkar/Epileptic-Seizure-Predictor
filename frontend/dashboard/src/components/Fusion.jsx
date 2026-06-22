import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { motion } from 'framer-motion'
import fusion from '../data/fusion.json'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'
import FusionSimulator from './FusionSimulator'

function MiniConfusion({ title, data, accent }) {
  const cells = [
    { label: 'TN', value: data.tn, good: true },
    { label: 'FP', value: data.fp, good: false },
    { label: 'FN', value: data.fn, good: false },
    { label: 'TP', value: data.tp, good: true },
  ]
  return (
    <div>
      <div className="text-sm font-medium text-slate-300 mb-2">{title}</div>
      <div className="grid grid-cols-2 gap-1.5">
        {cells.map((c) => (
          <div
            key={c.label}
            className="rounded-lg p-3 text-center"
            style={{
              background: c.good ? `${accent}1a` : 'rgba(239,68,68,0.12)',
              border: `1px solid ${c.good ? accent + '44' : 'rgba(239,68,68,0.3)'}`,
            }}
          >
            <div className="stat-num text-lg text-white">{c.value.toLocaleString()}</div>
            <div className="text-[11px] text-slate-500">{c.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Fusion() {
  return (
    <section id="fusion" className="section-pad">
      <SectionHeader
        eyebrow="06 · The Fusion Layer"
        title="Where the two minds meet"
        subtitle="An attention-gated layer learns a per-patient weight α, blending the EEG embedding with the genetic risk into one calibrated score."
        accentDot="#34d399"
      />

      {/* attention diagram */}
      <Reveal>
        <div className="mt-12 glass p-6 sm:p-8">
          <div className="grid md:grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-4">
            <div className="space-y-3">
              <div className="rounded-xl bg-eeg/10 border border-eeg/30 p-4 text-center">
                <div className="text-eeg font-semibold text-sm">EEG embedding</div>
                <div className="text-xs text-slate-500 mt-0.5 stat-num">512-dim</div>
              </div>
              <div className="rounded-xl bg-gene/10 border border-gene/30 p-4 text-center">
                <div className="text-gene font-semibold text-sm">Genetic score</div>
                <div className="text-xs text-slate-500 mt-0.5 stat-num">1-dim</div>
              </div>
            </div>

            <Arrow />

            <div className="rounded-xl bg-fuse/10 border border-fuse/30 p-5 text-center relative overflow-hidden">
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-fuse/10 to-transparent"
                animate={{ x: ['-100%', '100%'] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
              />
              <div className="relative">
                <div className="text-fuse font-semibold">Attention Gate</div>
                <div className="text-xs text-slate-400 mt-1">α = σ(W_e·h_eeg + W_g·h_gen + b)</div>
                <div className="mt-2 text-xs text-slate-500">projects both to 128-dim, gates, fuses</div>
              </div>
            </div>

            <Arrow />

            <div className="rounded-xl bg-gradient-to-br from-fuse/15 to-eeg/10 border border-white/10 p-5 text-center">
              <div className="text-white font-semibold">Risk Head</div>
              <div className="text-xs text-slate-400 mt-1">→ P_final ∈ [0,1]</div>
              <div className="mt-2 flex gap-1 justify-center">
                {fusion.alert_levels.map((l) => (
                  <span key={l.level} className="h-2 w-5 rounded-full" style={{ background: l.color }} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </Reveal>

      {/* simulator */}
      <Reveal>
        <div className="mt-6">
          <FusionSimulator />
        </div>
      </Reveal>

      {/* comparison */}
      <div className="mt-6 grid lg:grid-cols-2 gap-4">
        <Reveal>
          <div className="glass p-6 h-full">
            <h4 className="font-semibold text-white">EEG-only vs Fusion</h4>
            <p className="text-sm text-slate-500 mb-4">Fusion wins on every metric · {fusion.test_samples.toLocaleString()} test windows.</p>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={fusion.comparison} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c2438" vertical={false} />
                <XAxis dataKey="metric" tick={{ fontSize: 10 }} stroke="#475569" angle={-30} textAnchor="end" height={50} interval={0} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#475569" />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="eeg" name="EEG-only" fill="#0e7490" radius={[3, 3, 0, 0]} />
                <Bar dataKey="fusion" name="Fusion" fill="#34d399" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="glass p-6 h-full flex flex-col">
            <h4 className="font-semibold text-white">Confusion: catching more, false-alarming less</h4>
            <p className="text-sm text-slate-500 mb-5">Fusion finds 55 more seizure windows and cuts 61 false positives.</p>
            <div className="grid grid-cols-2 gap-5">
              <MiniConfusion title="EEG-only" data={fusion.confusion.eeg} accent="#0e7490" />
              <MiniConfusion title="Fusion" data={fusion.confusion.fusion} accent="#34d399" />
            </div>
            <div className="mt-auto pt-5 grid grid-cols-3 gap-2 text-center">
              <Delta label="True positives" from={fusion.confusion.eeg.tp} to={fusion.confusion.fusion.tp} good />
              <Delta label="False positives" from={fusion.confusion.eeg.fp} to={fusion.confusion.fusion.fp} good={false} />
              <Delta label="Missed (FN)" from={fusion.confusion.eeg.fn} to={fusion.confusion.fusion.fn} good={false} />
            </div>
          </div>
        </Reveal>
      </div>

      {/* attention analysis */}
      <Reveal>
        <div className="mt-4 glass p-5 border-fuse/20 flex items-start gap-3">
          <span className="mt-0.5 grid place-items-center h-6 w-6 rounded-full bg-fuse/15 text-fuse text-xs shrink-0">α</span>
          <p className="text-sm text-slate-400">
            <span className="text-fuse font-medium">Learned attention α = {fusion.attention.mean.toFixed(3)} (σ ≈ 0).</span>{' '}
            The model settled on a near-fixed 50/50 split because the synthetic genetic scores have very low variance. The gains flow through the risk head, which also receives the raw genetic score directly.
          </p>
        </div>
      </Reveal>
    </section>
  )
}

function Arrow() {
  return (
    <div className="hidden md:flex justify-center">
      <svg width="32" height="16" viewBox="0 0 32 16" className="text-fuse/60">
        <path d="M0 8 H26 M22 4 L28 8 L22 12" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}

function Delta({ label, from, to, good }) {
  const diff = to - from
  const improved = good ? diff > 0 : diff < 0
  return (
    <div className="rounded-lg bg-white/[0.03] border border-white/[0.05] p-2.5">
      <div className={`stat-num text-sm ${improved ? 'text-fuse' : 'text-red-400'}`}>
        {diff > 0 ? '+' : ''}{diff}
      </div>
      <div className="text-[11px] text-slate-500">{label}</div>
    </div>
  )
}
