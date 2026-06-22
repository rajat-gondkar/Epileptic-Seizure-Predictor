import { useState } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { motion } from 'framer-motion'
import preprocessing from '../data/preprocessing.json'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'
import EEGStrip from './ui/EEGStrip'

export default function Preprocessing() {
  const [filtered, setFiltered] = useState(false)

  return (
    <section id="preprocessing" className="section-pad">
      <SectionHeader
        eyebrow="02 · EEG Preprocessing"
        title="From raw EDF to model-ready windows"
        subtitle="Every 30-second window is filtered, resampled, and normalised on the fly — no precomputed feature extraction."
        accentDot="#22d3ee"
      />

      {/* pipeline steps */}
      <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {preprocessing.steps.map((s, i) => (
          <Reveal key={s.title} delay={i * 0.06}>
            <div className="glass glass-hover p-5 h-full relative overflow-hidden">
              <div className="absolute -right-2 -top-3 text-6xl font-bold text-white/[0.03] stat-num">
                {String(i + 1).padStart(2, '0')}
              </div>
              <div className="text-eeg text-sm font-mono">step {i + 1}</div>
              <h3 className="mt-1 font-semibold text-white">{s.title}</h3>
              <p className="mt-1.5 text-sm text-slate-400 leading-relaxed">{s.desc}</p>
            </div>
          </Reveal>
        ))}
      </div>

      <div className="mt-8 grid lg:grid-cols-5 gap-4">
        {/* waveform demo */}
        <Reveal className="lg:col-span-3">
          <div className="glass p-6 h-full">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <h4 className="font-semibold text-white">Bandpass filtering, live</h4>
                <p className="text-sm text-slate-500">0.5–45 Hz Butterworth removes drift and noise.</p>
              </div>
              <div className="flex items-center gap-1 rounded-lg bg-white/[0.04] p-1">
                <button
                  onClick={() => setFiltered(false)}
                  className={`px-3 py-1.5 rounded-md text-sm transition-colors ${!filtered ? 'bg-eeg/20 text-eeg' : 'text-slate-400'}`}
                >
                  Raw signal
                </button>
                <button
                  onClick={() => setFiltered(true)}
                  className={`px-3 py-1.5 rounded-md text-sm transition-colors ${filtered ? 'bg-eeg/20 text-eeg' : 'text-slate-400'}`}
                >
                  Filtered
                </button>
              </div>
            </div>
            <div className="mt-4 rounded-xl bg-ink-950/60 border border-white/[0.05] p-2">
              <EEGStrip filtered={filtered} height={240} />
            </div>
            <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500">
              <span>Input shape: <span className="text-slate-300 stat-num">(16, 3840, 19)</span></span>
              <span>30s × 128 Hz = 3840 timesteps</span>
              <span>{filtered ? '0.5–45 Hz bandpass · z-scored' : 'unfiltered · drift + line noise + artifacts'}</span>
            </div>
          </div>
        </Reveal>

        {/* class distribution */}
        <Reveal delay={0.1} className="lg:col-span-2">
          <div className="glass p-6 h-full">
            <h4 className="font-semibold text-white">Window class distribution</h4>
            <p className="text-sm text-slate-500">The core challenge: extreme imbalance.</p>
            <ResponsiveContainer width="100%" height={210}>
              <PieChart>
                <Pie
                  data={preprocessing.class_distribution}
                  dataKey="count"
                  nameKey="name"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={3}
                  stroke="none"
                >
                  {preprocessing.class_distribution.map((c) => (
                    <Cell key={c.name} fill={c.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(v, n) => [`${v.toLocaleString()} windows`, n]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2">
              {preprocessing.class_distribution.map((c) => (
                <div key={c.name} className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: c.color }} />
                    {c.name}
                  </span>
                  <span className="text-slate-400 stat-num">{c.pct}% · {c.count.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </div>

      {/* 3-zone labeling */}
      <Reveal>
        <div className="mt-8 glass p-6">
          <h4 className="font-semibold text-white">Three-zone seizure labeling</h4>
          <p className="text-sm text-slate-500 mb-5">Each seizure is modelled with a preictal window and a prediction-horizon gap.</p>
          <div className="flex flex-col sm:flex-row gap-1.5">
            {preprocessing.zones.map((z, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scaleX: 0.6 }}
                whileInView={{ opacity: 1, scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.12, duration: 0.5 }}
                style={{ originX: 0, background: `${z.color}1a`, borderColor: `${z.color}55` }}
                className="flex-1 rounded-lg p-4 border"
              >
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: z.color }} />
                  <span className="font-medium text-white text-sm">{z.zone}</span>
                  <span className="text-[10px] text-slate-500 font-mono">label {z.label}</span>
                </div>
                <p className="mt-1.5 text-xs text-slate-400">{z.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </Reveal>
    </section>
  )
}
