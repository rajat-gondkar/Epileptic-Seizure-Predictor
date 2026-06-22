import fusion from '../data/fusion.json'
import project from '../data/project.json'
import SectionHeader from './ui/SectionHeader'
import Reveal from './ui/Reveal'
import CountUp from './ui/CountUp'

export default function Results() {
  return (
    <section id="results" className="section-pad">
      <SectionHeader
        eyebrow="08 · Outcome"
        title="What we achieved"
        subtitle="A complete, working multimodal pipeline — from raw signals to a calibrated clinical alert."
        accentDot="#34d399"
        align="center"
      />

      {/* scoreboard */}
      <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {project.headline_stats.map((s, i) => {
          const decimals = s.format === 'percent2' ? 2 : s.format === 'percent1' ? 1 : 3
          const prefix = s.format === 'percent1' ? '+' : ''
          const suffix = s.format?.startsWith('percent') ? '%' : ''
          const accentColor = { eeg: '#22d3ee', gene: '#a78bfa', fuse: '#34d399' }[s.accent]
          return (
            <Reveal key={s.label} delay={i * 0.08}>
              <div className="glass p-6 text-center h-full">
                <div className="stat-num text-4xl" style={{ color: accentColor }}>
                  <CountUp to={s.value} decimals={decimals} prefix={prefix} suffix={suffix} />
                </div>
                <div className="mt-2 text-sm text-slate-300">{s.label}</div>
                <div className="text-xs text-slate-500">{s.note}</div>
              </div>
            </Reveal>
          )
        })}
      </div>

      <div className="mt-6 grid lg:grid-cols-2 gap-4">
        <Reveal>
          <div className="glass p-6 h-full">
            <h4 className="font-semibold text-fuse">Key findings</h4>
            <ul className="mt-4 space-y-3">
              {fusion.key_findings.map((f, i) => (
                <li key={i} className="flex gap-3 text-sm text-slate-300">
                  <svg className="shrink-0 mt-0.5 text-fuse" width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8.5l3 3 7-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="glass p-6 h-full border-amber-500/15">
            <h4 className="font-semibold text-amber-400">Honest limitations</h4>
            <ul className="mt-4 space-y-3">
              {fusion.limitations.map((f, i) => (
                <li key={i} className="flex gap-3 text-sm text-slate-300">
                  <svg className="shrink-0 mt-0.5 text-amber-400" width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M8 1.5l6.5 11.5h-13z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                    <path d="M8 6.5v3M8 11.2v.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
