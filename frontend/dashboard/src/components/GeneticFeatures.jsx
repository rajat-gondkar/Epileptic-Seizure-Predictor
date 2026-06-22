import { useState } from 'react'
import { motion } from 'framer-motion'
import genetic from '../data/genetic.json'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'

export default function GeneticFeatures() {
  const [hovered, setHovered] = useState(null)
  const active = hovered != null ? genetic.feature_vector[hovered] : null

  return (
    <section id="genetics" className="section-pad">
      <SectionHeader
        eyebrow="03 · Genetic Feature Engineering"
        title="A 22-dimensional genetic fingerprint"
        subtitle="Each patient is encoded as a single vector of weighted mutation flags, constraint scores, polygenic risk, and engineered interactions."
        accentDot="#a78bfa"
      />

      <div className="mt-12 grid lg:grid-cols-5 gap-4">
        {/* interactive vector */}
        <Reveal className="lg:col-span-3">
          <div className="glass p-6 h-full">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-white">The feature vector</h4>
              <span className="text-xs text-slate-500">hover a cell</span>
            </div>

            <div className="mt-4 grid grid-cols-11 gap-1.5">
              {genetic.feature_vector.map((f, i) => {
                const color = genetic.groups[f.group].color
                const isActive = hovered === i
                return (
                  <motion.button
                    key={f.idx}
                    onMouseEnter={() => setHovered(i)}
                    onFocus={() => setHovered(i)}
                    onMouseLeave={() => setHovered(null)}
                    initial={{ opacity: 0, scale: 0.6 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.02, duration: 0.3 }}
                    className="aspect-square rounded-md grid place-items-center text-[10px] font-mono transition-all"
                    style={{
                      background: isActive ? color : `${color}26`,
                      color: isActive ? '#0a0e1a' : color,
                      boxShadow: isActive ? `0 0 16px ${color}` : 'none',
                      transform: isActive ? 'scale(1.12)' : 'scale(1)',
                    }}
                  >
                    {f.idx}
                  </motion.button>
                )
              })}
            </div>

            {/* legend */}
            <div className="mt-4 flex flex-wrap gap-3">
              {Object.entries(genetic.groups).map(([key, g]) => (
                <span key={key} className="flex items-center gap-1.5 text-xs text-slate-400">
                  <span className="h-2.5 w-2.5 rounded" style={{ background: g.color }} />
                  {g.label}
                </span>
              ))}
            </div>

            {/* detail */}
            <div className="mt-4 rounded-xl bg-ink-950/60 border border-white/[0.05] p-4 min-h-[78px]">
              {active ? (
                <motion.div
                  key={active.idx}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs font-mono px-2 py-0.5 rounded"
                      style={{ background: `${genetic.groups[active.group].color}26`, color: genetic.groups[active.group].color }}
                    >
                      [{active.idx}]
                    </span>
                    <span className="font-mono text-sm text-white">{active.name}</span>
                  </div>
                  <p className="mt-1.5 text-sm text-slate-400">{active.desc}</p>
                </motion.div>
              ) : (
                <p className="text-sm text-slate-500">
                  9 mutation flags · 7 pLI scores · 1 polygenic risk score · 5 interaction features = 22 dimensions
                </p>
              )}
            </div>
          </div>
        </Reveal>

        {/* tiers */}
        <Reveal delay={0.1} className="lg:col-span-2">
          <div className="glass p-6 h-full">
            <h4 className="font-semibold text-white">Gene risk tiers</h4>
            <p className="text-sm text-slate-500 mb-4">Literature-backed weights (Brunklaus 2022, Thomas 2019).</p>
            <div className="space-y-3">
              {genetic.tiers.map((t) => (
                <div key={t.tier} className="rounded-xl border p-3.5" style={{ borderColor: `${t.color}40`, background: `${t.color}0d` }}>
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-sm" style={{ color: t.color }}>{t.tier}</span>
                    <span className="text-[11px] text-slate-500">{t.relevance}</span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {t.genes.map((g) => (
                      <span key={g.g} className="text-xs font-mono px-2 py-1 rounded-md bg-white/[0.05] text-slate-300">
                        {g.g} <span className="text-slate-500">·{g.w.toFixed(2)}</span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </div>

      <Reveal>
        <div className="mt-4 glass p-5 border-gene/20 flex items-start gap-3">
          <span className="mt-0.5 grid place-items-center h-6 w-6 rounded-full bg-gene/15 text-gene text-xs shrink-0">i</span>
          <p className="text-sm text-slate-400">
            <span className="text-gene font-medium">{genetic.cohort.patients.toLocaleString()} synthetic patients.</span>{' '}
            {genetic.cohort.note}
          </p>
        </div>
      </Reveal>
    </section>
  )
}
