import { motion } from 'framer-motion'
import project from '../data/project.json'
import { formatValue, accentMap } from '../lib/format'
import EEGWave from './ui/EEGWave'
import CountUp from './ui/CountUp'

function HeadlineStat({ stat, i }) {
  const accent = accentMap[stat.accent]
  const decimals = stat.format === 'percent2' ? 2 : stat.format === 'percent1' ? 1 : 3
  const prefix = stat.format === 'percent1' ? '+' : ''
  const suffix = stat.format?.startsWith('percent') ? '%' : ''
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 + i * 0.12, duration: 0.6 }}
      className={`glass glass-hover p-5 ${accent.border}`}
    >
      <div className={`text-3xl sm:text-4xl stat-num ${accent.text}`}>
        <CountUp to={stat.value} decimals={decimals} prefix={prefix} suffix={suffix} />
      </div>
      <div className="mt-1 text-sm font-medium text-slate-200">{stat.label}</div>
      <div className="text-xs text-slate-500">{stat.note}</div>
    </motion.div>
  )
}

export default function Hero() {
  return (
    <section id="hero" className="relative min-h-screen flex items-center pt-20 overflow-hidden">
      {/* ambient waves */}
      <div className="absolute inset-0 opacity-[0.15] pointer-events-none">
        <div className="absolute top-[18%] w-full">
          <EEGWave color="#22d3ee" filtered height={180} lines={3} speed={0.7} />
        </div>
        <div className="absolute bottom-[12%] w-full">
          <EEGWave color="#a78bfa" filtered height={140} lines={2} speed={0.5} />
        </div>
      </div>

      <div className="section-pad relative w-full">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <motion.span
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="eyebrow"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-fuse" style={{ boxShadow: '0 0 12px #34d399' }} />
              {project.institution}
            </motion.span>

            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.7 }}
              className="mt-5 text-5xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-[1.02] text-white"
            >
              EEG<span className="text-slate-600"> × </span>
              <span className="grad-text">Genetic</span>
              <br />
              Fusion
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25, duration: 0.7 }}
              className="mt-6 text-lg text-slate-400 max-w-xl leading-relaxed"
            >
              {project.tagline}
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.7 }}
              className="mt-8 flex flex-wrap gap-3"
            >
              <a
                href="#fusion"
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-eeg to-fuse text-ink-950 font-semibold hover:opacity-90 transition-opacity"
              >
                Launch Fusion Simulator
              </a>
              <a
                href="#datasets"
                className="px-6 py-3 rounded-xl glass glass-hover font-medium text-slate-200"
              >
                Explore the Pipeline
              </a>
            </motion.div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {project.headline_stats.map((s, i) => (
              <HeadlineStat key={s.label} stat={s} i={i} />
            ))}
          </div>
        </div>

        {/* architecture strip */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.7 }}
          className="mt-16 glass p-6 sm:p-8"
        >
          <div className="flex flex-col md:flex-row items-stretch gap-4 md:gap-2">
            <Branch title="EEG Branch" color="eeg" tag="BiLSTM + Attention" sub="3840 × 19 → 512-dim embedding" />
            <Connector />
            <Branch title="Genetic Branch" color="gene" tag="XGBoost" sub="22-dim genetic vector → risk" />
            <Connector />
            <div className="flex-1 grid place-items-center rounded-xl bg-fuse/10 border border-fuse/30 p-4 text-center">
              <div>
                <div className="text-fuse font-semibold">Attention-Gated Fusion</div>
                <div className="text-xs text-slate-400 mt-1">α · EEG + (1−α) · Genetic → P<sub>final</sub></div>
              </div>
            </div>
            <Connector />
            <div className="flex-1 grid place-items-center rounded-xl bg-gradient-to-br from-fuse/10 to-eeg/10 border border-white/10 p-4 text-center">
              <div>
                <div className="text-white font-semibold">4-Level Alert</div>
                <div className="flex gap-1 mt-2 justify-center">
                  {['#34d399', '#facc15', '#fb923c', '#ef4444'].map((c) => (
                    <span key={c} className="h-2.5 w-6 rounded-full" style={{ background: c }} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function Branch({ title, color, tag, sub }) {
  const accent = accentMap[color]
  return (
    <div className={`flex-1 rounded-xl ${accent.bg} border ${accent.border} p-4 text-center`}>
      <div className={`font-semibold ${accent.text}`}>{title}</div>
      <div className="text-sm text-slate-300 mt-1">{tag}</div>
      <div className="text-xs text-slate-500 mt-1">{sub}</div>
    </div>
  )
}

function Connector() {
  return (
    <div className="hidden md:flex items-center justify-center px-1">
      <svg width="28" height="16" viewBox="0 0 28 16" className="text-slate-600">
        <path d="M0 8 H22 M18 4 L24 8 L18 12" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}
