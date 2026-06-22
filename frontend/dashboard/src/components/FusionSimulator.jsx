import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import fusion from '../data/fusion.json'
import { alertForScore } from '../lib/format'
import RiskGauge from './RiskGauge'

function Slider({ label, value, onChange, color, hint }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium text-slate-200 flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
          {label}
        </label>
        <span className="stat-num text-sm" style={{ color }}>{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full appearance-none cursor-pointer h-2 rounded-full"
        style={{
          color,
          background: `linear-gradient(to right, ${color} 0%, ${color} ${value * 100}%, rgba(255,255,255,0.08) ${value * 100}%, rgba(255,255,255,0.08) 100%)`,
        }}
      />
      <p className="mt-1.5 text-xs text-slate-500">{hint}</p>
    </div>
  )
}

export default function FusionSimulator() {
  const [eeg, setEeg] = useState(0.62)
  const [gene, setGene] = useState(0.48)
  const [alpha, setAlpha] = useState(0.5)

  // Documented fusion blend: P_final = α·P_eeg + (1−α)·P_genetic
  const pFinal = alpha * eeg + (1 - alpha) * gene
  const alert = alertForScore(pFinal, fusion.alert_levels)

  return (
    <div className="glass p-6 sm:p-8 border-fuse/20 shadow-glow-emerald">
      <div className="grid lg:grid-cols-2 gap-8 items-center">
        {/* controls */}
        <div className="space-y-6">
          <div>
            <h4 className="text-lg font-semibold text-white">Interactive fusion simulator</h4>
            <p className="text-sm text-slate-400">
              Adjust each branch's prediction and the attention gate to see how the fused risk score and clinical alert respond.
            </p>
          </div>

          <Slider
            label="EEG branch · preictal probability"
            value={eeg}
            onChange={setEeg}
            color="#22d3ee"
            hint="Output of the BiLSTM + attention model on a 30s window."
          />
          <Slider
            label="Genetic branch · risk score"
            value={gene}
            onChange={setGene}
            color="#a78bfa"
            hint="XGBoost probability from the 22-dim genetic vector."
          />

          <div>
            <Slider
              label="Attention gate · α (trust in EEG)"
              value={alpha}
              onChange={setAlpha}
              color="#34d399"
              hint="P_final = α · EEG + (1 − α) · Genetic"
            />
            <button
              onClick={() => setAlpha(0.5)}
              className="mt-2 text-xs px-3 py-1.5 rounded-md bg-fuse/10 text-fuse border border-fuse/20 hover:bg-fuse/20 transition-colors"
            >
              Reset α to learned value (0.50)
            </button>
          </div>
        </div>

        {/* output */}
        <div className="flex flex-col items-center">
          <RiskGauge value={pFinal} color={alert.color} levels={fusion.alert_levels} />

          <AnimatePresence mode="wait">
            <motion.div
              key={alert.level}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
              className="mt-2 text-center"
            >
              <div
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl font-semibold"
                style={{ background: `${alert.color}1f`, color: alert.color, border: `1px solid ${alert.color}55` }}
              >
                <span className="h-2.5 w-2.5 rounded-full animate-pulse" style={{ background: alert.color, boxShadow: `0 0 12px ${alert.color}` }} />
                Level {alert.level} · {alert.name} Alert
              </div>
              <p className="mt-2 text-sm text-slate-400">{alert.desc}</p>
            </motion.div>
          </AnimatePresence>

          {/* contribution breakdown */}
          <div className="mt-5 w-full grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-eeg/10 border border-eeg/20 p-3 text-center">
              <div className="text-xs text-slate-400">EEG contribution</div>
              <div className="stat-num text-lg text-eeg">{(alpha * eeg).toFixed(3)}</div>
              <div className="text-[11px] text-slate-500">{(alpha * 100).toFixed(0)}% weight</div>
            </div>
            <div className="rounded-xl bg-gene/10 border border-gene/20 p-3 text-center">
              <div className="text-xs text-slate-400">Genetic contribution</div>
              <div className="stat-num text-lg text-gene">{((1 - alpha) * gene).toFixed(3)}</div>
              <div className="text-[11px] text-slate-500">{((1 - alpha) * 100).toFixed(0)}% weight</div>
            </div>
          </div>
        </div>
      </div>

      {/* alert legend */}
      <div className="mt-8 pt-6 border-t border-white/[0.06] grid grid-cols-2 sm:grid-cols-4 gap-2">
        {fusion.alert_levels.map((lvl) => {
          const isActive = alert.level === lvl.level
          return (
            <div
              key={lvl.level}
              className="rounded-lg p-3 transition-all"
              style={{
                background: isActive ? `${lvl.color}1f` : 'rgba(255,255,255,0.02)',
                border: `1px solid ${isActive ? lvl.color + '66' : 'rgba(255,255,255,0.05)'}`,
              }}
            >
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: lvl.color }} />
                <span className="text-sm font-medium text-white">{lvl.name}</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 stat-num">
                {lvl.level === 1 ? '0.00' : fusion.alert_levels[lvl.level - 2].max.toFixed(2)}
                {' – '}
                {lvl.level === 4 ? '1.00' : lvl.max.toFixed(2)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
