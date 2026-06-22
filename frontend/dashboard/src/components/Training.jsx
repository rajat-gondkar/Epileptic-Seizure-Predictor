import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
  RadialBarChart, RadialBar, PolarAngleAxis, BarChart, Bar, Cell,
} from 'recharts'
import { motion, AnimatePresence } from 'framer-motion'
import training from '../data/training.json'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'
import ConfusionMatrix from './ConfusionMatrix'

function ConfigGrid({ items, accent }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {items.map((c) => (
        <div key={c.k} className="rounded-lg bg-white/[0.03] border border-white/[0.05] px-3 py-2">
          <div className="stat-num text-sm" style={{ color: accent }}>{c.v}</div>
          <div className="text-[11px] text-slate-500">{c.k}</div>
        </div>
      ))}
    </div>
  )
}

function EEGPanel() {
  const { eeg } = training
  return (
    <div className="space-y-4">
      <div className="glass p-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h4 className="font-semibold text-white">{eeg.architecture}</h4>
            <p className="text-sm text-slate-500">{eeg.params.toLocaleString()} parameters · trained 15 epochs</p>
          </div>
        </div>
        <div className="mt-4">
          <ConfigGrid items={eeg.config} accent="#22d3ee" />
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="glass p-6">
          <h4 className="font-semibold text-white">Training vs validation loss</h4>
          <p className="text-sm text-slate-500 mb-4">Stable validation loss — generalises without collapse.</p>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={eeg.loss_curve} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c2438" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="#475569" />
              <YAxis tick={{ fontSize: 11 }} stroke="#475569" />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="train" name="Train" stroke="#22d3ee" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="val" name="Validation" stroke="#f59e0b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="glass p-6">
          <h4 className="font-semibold text-white">Confusion matrix</h4>
          <p className="text-sm text-slate-500 mb-4">53,858 test windows · {eeg.accuracy}% accuracy.</p>
          <div className="overflow-x-auto">
            <ConfusionMatrix labels={eeg.confusion.labels} matrix={eeg.confusion.matrix} accent="#22d3ee" />
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="glass p-6">
          <h4 className="font-semibold text-white">Per-class metrics</h4>
          <p className="text-sm text-slate-500 mb-4">Precision, recall and F1 per class.</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={eeg.per_class} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c2438" vertical={false} />
              <XAxis dataKey="class" tick={{ fontSize: 11 }} stroke="#475569" />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#475569" />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="precision" name="Precision" fill="#22d3ee" radius={[3, 3, 0, 0]} />
              <Bar dataKey="recall" name="Recall" fill="#34d399" radius={[3, 3, 0, 0]} />
              <Bar dataKey="f1" name="F1" fill="#a78bfa" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass p-6">
          <h4 className="font-semibold text-white">Highlights</h4>
          <div className="mt-4 grid grid-cols-2 gap-3">
            {eeg.highlights.map((h) => (
              <div key={h.label} className="rounded-xl bg-white/[0.03] border border-white/[0.05] p-4">
                <div className="stat-num text-2xl text-eeg">{h.value}</div>
                <div className="text-xs text-slate-400 mt-1">{h.label}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {eeg.roc_auc.map((r) => (
              <span key={r.class} className="text-xs px-2.5 py-1 rounded-md bg-eeg/10 text-eeg">
                ROC-AUC {r.class}: <span className="stat-num">{r.auc.toFixed(2)}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function GeneticPanel() {
  const { genetic } = training
  const radialData = genetic.metrics.map((m) => ({
    name: m.name,
    value: m.value * 100,
    fill: ['#a78bfa', '#c084fc', '#818cf8', '#e879f9', '#a78bfa', '#7c3aed'][genetic.metrics.indexOf(m) % 6],
  }))
  return (
    <div className="space-y-4">
      <div className="glass p-6">
        <h4 className="font-semibold text-white">{genetic.architecture}</h4>
        <p className="text-sm text-slate-500 mb-4">Gradient-boosted trees on the 10k synthetic cohort.</p>
        <ConfigGrid items={genetic.config} accent="#a78bfa" />
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="glass p-6">
          <h4 className="font-semibold text-white">Performance metrics</h4>
          <p className="text-sm text-slate-500 mb-2">AUC-PR favoured over ROC for imbalanced data.</p>
          <ResponsiveContainer width="100%" height={280}>
            <RadialBarChart innerRadius="25%" outerRadius="100%" data={radialData} startAngle={90} endAngle={-270}>
              <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
              <RadialBar background={{ fill: '#1c2438' }} dataKey="value" cornerRadius={6} />
              <Tooltip formatter={(v) => `${(v / 100).toFixed(3)}`} />
            </RadialBarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-3 gap-2 mt-2">
            {genetic.metrics.map((m) => (
              <div key={m.name} className="text-center">
                <div className="stat-num text-sm text-gene">{m.value.toFixed(3)}</div>
                <div className="text-[11px] text-slate-500">{m.name}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass p-6">
          <h4 className="font-semibold text-white">SHAP feature importance</h4>
          <p className="text-sm text-slate-500 mb-4">What the model learned to rely on.</p>
          <div className="space-y-2">
            {genetic.shap_ranking.map((f, i) => (
              <motion.div
                key={f}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="flex items-center gap-3"
              >
                <span className="text-xs font-mono text-slate-500 w-4">{i + 1}</span>
                <div className="flex-1 h-8 rounded-lg bg-gene/10 border border-gene/20 flex items-center px-3"
                  style={{ width: `${100 - i * 8}%` }}>
                  <span className="text-sm text-slate-200">{f}</span>
                </div>
              </motion.div>
            ))}
          </div>
          <div className="mt-4 rounded-xl bg-white/[0.03] border border-white/[0.05] p-3">
            <p className="text-sm text-slate-400">{genetic.profile}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Training() {
  const [tab, setTab] = useState('eeg')
  return (
    <section id="training" className="section-pad">
      <SectionHeader
        eyebrow="05 · Model Training & Testing"
        title="Two branches, trained independently"
        subtitle="A BiLSTM learns temporal EEG patterns; XGBoost learns the genetic signal. Both are frozen before fusion."
        accentDot="#22d3ee"
      />

      <Reveal>
        <div className="mt-10 flex items-center gap-1 rounded-xl bg-white/[0.04] p-1 w-fit">
          <button
            onClick={() => setTab('eeg')}
            className={`px-5 py-2 rounded-lg text-sm font-medium transition-colors ${tab === 'eeg' ? 'bg-eeg/20 text-eeg' : 'text-slate-400 hover:text-white'}`}
          >
            EEG Branch · BiLSTM
          </button>
          <button
            onClick={() => setTab('genetic')}
            className={`px-5 py-2 rounded-lg text-sm font-medium transition-colors ${tab === 'genetic' ? 'bg-gene/20 text-gene' : 'text-slate-400 hover:text-white'}`}
          >
            Genetic Branch · XGBoost
          </button>
        </div>
      </Reveal>

      <div className="mt-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35 }}
          >
            {tab === 'eeg' ? <EEGPanel /> : <GeneticPanel />}
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  )
}
