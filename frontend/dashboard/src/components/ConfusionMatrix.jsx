import { useState } from 'react'
import { motion } from 'framer-motion'

/**
 * Square confusion matrix with row-normalised colour intensity.
 * matrix[i][j] = count of true class i predicted as class j.
 */
export default function ConfusionMatrix({ labels, matrix, accent = '#22d3ee' }) {
  const [hover, setHover] = useState(null)
  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0))

  return (
    <div className="inline-block">
      <div className="flex">
        <div className="w-20" />
        <div className="flex-1 text-center text-[11px] text-slate-500 pb-1">Predicted</div>
      </div>
      <div className="flex items-stretch">
        <div className="flex items-center">
          <span className="text-[11px] text-slate-500 -rotate-90 whitespace-nowrap w-5">True</span>
        </div>
        <div>
          {/* header */}
          <div className="flex">
            <div className="w-16" />
            {labels.map((l) => (
              <div key={l} className="w-[72px] text-center text-[11px] text-slate-400 pb-1 truncate px-0.5">{l}</div>
            ))}
          </div>
          {matrix.map((row, i) => (
            <div key={i} className="flex items-center">
              <div className="w-16 text-right pr-2 text-[11px] text-slate-400 truncate">{labels[i]}</div>
              {row.map((val, j) => {
                const frac = rowSums[i] ? val / rowSums[i] : 0
                const isDiag = i === j
                const base = isDiag ? accent : '#ef4444'
                const isHover = hover === `${i}-${j}`
                return (
                  <motion.div
                    key={j}
                    onMouseEnter={() => setHover(`${i}-${j}`)}
                    onMouseLeave={() => setHover(null)}
                    initial={{ opacity: 0, scale: 0.8 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: (i * row.length + j) * 0.04 }}
                    className="w-[72px] h-[72px] m-0.5 rounded-lg grid place-items-center cursor-default relative"
                    style={{
                      background: `${base}${Math.round((0.12 + frac * 0.78) * 255).toString(16).padStart(2, '0')}`,
                      outline: isHover ? `1.5px solid ${base}` : 'none',
                    }}
                  >
                    <div className="text-center">
                      <div className="stat-num text-sm text-white">{val.toLocaleString()}</div>
                      <div className="text-[10px] text-white/70">{(frac * 100).toFixed(1)}%</div>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
