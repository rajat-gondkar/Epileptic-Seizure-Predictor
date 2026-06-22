import { motion } from 'framer-motion'
import ctgan from '../data/ctgan.json'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'

function KSBar({ item, i }) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1.5">
        <span className="text-slate-300">{item.name}</span>
        <span className="stat-num" style={{ color: item.color }}>{item.value}%</span>
      </div>
      <div className="h-2.5 rounded-full bg-white/[0.05] overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${item.value}%` }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.15, duration: 1, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ background: item.color, boxShadow: `0 0 12px ${item.color}` }}
        />
      </div>
    </div>
  )
}

export default function Ctgan() {
  return (
    <section id="ctgan" className="section-pad">
      <SectionHeader
        eyebrow="04 · Synthetic Data (CTGAN)"
        title="Synthetic augmentation, validated honestly"
        subtitle="A conditional tabular GAN generates plausible records — used strictly as a validation benchmark, never to train the fusion model."
        accentDot="#a78bfa"
      />

      <div className="mt-12 grid lg:grid-cols-2 gap-4">
        <Reveal>
          <div className="glass p-6 h-full">
            <h4 className="font-semibold text-white">Generation pipeline</h4>
            <ol className="mt-4 space-y-3">
              {ctgan.pipeline.map((step, i) => (
                <motion.li
                  key={i}
                  initial={{ opacity: 0, x: -12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.08 }}
                  className="flex gap-3 text-sm"
                >
                  <span className="shrink-0 grid place-items-center h-6 w-6 rounded-full bg-gene/15 text-gene text-xs font-mono">
                    {i + 1}
                  </span>
                  <span className="text-slate-300 pt-0.5">{step}</span>
                </motion.li>
              ))}
            </ol>
          </div>
        </Reveal>

        <div className="space-y-4">
          <Reveal delay={0.1}>
            <div className="glass p-6">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-white">Kolmogorov–Smirnov pass rates</h4>
                <span className="text-xs text-slate-500">{ctgan.samples.toLocaleString()} samples</span>
              </div>
              <div className="mt-5 space-y-4">
                {ctgan.ks_results.map((r, i) => (
                  <KSBar key={r.name} item={r} i={i} />
                ))}
              </div>
            </div>
          </Reveal>

          <Reveal delay={0.2}>
            <div className="glass p-6 border-gene/20">
              <h4 className="font-semibold text-white text-sm">Why this is the right call</h4>
              <p className="mt-2 text-sm text-slate-400 leading-relaxed">{ctgan.interpretation}</p>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}
