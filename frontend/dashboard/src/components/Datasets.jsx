import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid, Legend,
} from 'recharts'
import datasets from '../data/datasets.json'
import { accentMap } from '../lib/format'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'

function DatasetCard({ card, i }) {
  const accent = accentMap[card.accent]
  return (
    <Reveal delay={i * 0.08}>
      <div className={`glass glass-hover p-6 h-full ${accent.border}`}>
        <div className="flex items-start justify-between">
          <h3 className="text-lg font-semibold text-white">{card.name}</h3>
          <span className={`text-[10px] uppercase tracking-wider px-2 py-1 rounded-md ${accent.bg} ${accent.text}`}>
            {card.source}
          </span>
        </div>
        <p className="mt-2 text-sm text-slate-400 leading-relaxed">{card.desc}</p>
        <div className="mt-4 grid grid-cols-2 gap-2">
          {card.stats.map((s) => (
            <div key={s.k} className="rounded-lg bg-white/[0.03] border border-white/[0.05] px-3 py-2">
              <div className={`stat-num text-sm ${accent.text}`}>{s.v}</div>
              <div className="text-[11px] text-slate-500">{s.k}</div>
            </div>
          ))}
        </div>
      </div>
    </Reveal>
  )
}

export default function Datasets() {
  return (
    <section id="datasets" className="section-pad">
      <SectionHeader
        eyebrow="01 · Data Foundation"
        title="Four datasets, two modalities"
        subtitle="The system fuses scalp EEG recordings with curated genetic evidence from three public genomic databases."
        accentDot="#22d3ee"
      />

      <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {datasets.cards.map((c, i) => (
          <DatasetCard key={c.name} card={c} i={i} />
        ))}
      </div>

      <div className="mt-8 grid lg:grid-cols-2 gap-4">
        <Reveal>
          <div className="glass p-6">
            <h4 className="font-semibold text-white">CHB-MIT files per patient</h4>
            <p className="text-sm text-slate-500 mb-4">Total EDF files vs. files containing seizure annotations.</p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={datasets.patients} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c2438" vertical={false} />
                <XAxis dataKey="id" tick={{ fontSize: 11 }} stroke="#475569" interval={0} angle={-35} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} stroke="#475569" />
                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="files" name="EDF files" fill="#22d3ee" radius={[3, 3, 0, 0]} />
                <Bar dataKey="seizureFiles" name="Seizure files" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="glass p-6">
            <h4 className="font-semibold text-white">Pathogenic variants per gene</h4>
            <p className="text-sm text-slate-500 mb-4">13,331 ClinVar pathogenic variants across 9 epilepsy genes.</p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={datasets.geneVariants} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c2438" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} stroke="#475569" />
                <YAxis type="category" dataKey="gene" tick={{ fontSize: 11 }} stroke="#475569" width={60} />
                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                <Bar dataKey="variants" name="Pathogenic variants" radius={[0, 3, 3, 0]}>
                  {datasets.geneVariants.map((g, idx) => (
                    <Cell key={idx} fill={`rgba(167, 139, 250, ${0.45 + (g.pLI > 0.9 ? 0.5 : 0.1)})`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
