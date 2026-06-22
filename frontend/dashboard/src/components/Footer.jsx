import project from '../data/project.json'

export default function Footer() {
  return (
    <footer className="border-t border-white/[0.06] mt-10">
      <div className="max-w-7xl mx-auto px-5 sm:px-8 py-12">
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="grid place-items-center h-8 w-8 rounded-lg bg-gradient-to-br from-eeg/20 to-gene/20 border border-white/10">
                <span className="h-2 w-2 rounded-full bg-gradient-to-r from-eeg to-gene" />
              </span>
              <span className="font-semibold text-white">Neuro<span className="grad-text">Genix</span></span>
            </div>
            <p className="mt-3 text-sm text-slate-500 max-w-xs">{project.subtitle}</p>
          </div>

          <div>
            <h5 className="text-sm font-semibold text-slate-300">Team</h5>
            <ul className="mt-3 space-y-1.5">
              {project.team.map((m) => (
                <li key={m} className="text-sm text-slate-400">{m}</li>
              ))}
            </ul>
            <p className="mt-3 text-sm text-slate-500">Guide: <span className="text-slate-400">{project.guide}</span></p>
          </div>

          <div>
            <h5 className="text-sm font-semibold text-slate-300">Built with</h5>
            <div className="mt-3 flex flex-wrap gap-2">
              {['React', 'Vite', 'Tailwind CSS', 'Recharts', 'Framer Motion'].map((t) => (
                <span key={t} className="text-xs px-2.5 py-1 rounded-md bg-white/[0.04] border border-white/[0.06] text-slate-400">
                  {t}
                </span>
              ))}
            </div>
            <p className="mt-4 text-sm text-slate-500">{project.institution}</p>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-white/[0.06] text-center text-xs text-slate-600">
          EEG–Genetic Fusion for Epileptic Seizure Prediction · Showcase dashboard
        </div>
      </div>
    </footer>
  )
}
