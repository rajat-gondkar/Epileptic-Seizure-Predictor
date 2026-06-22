import { motion } from 'framer-motion'

/**
 * Semicircular risk gauge. `value` in [0,1]. `levels` defines colour bands.
 */
export default function RiskGauge({ value, color, levels }) {
  const size = 260
  const stroke = 22
  const r = (size - stroke) / 2
  const cx = size / 2
  const cy = size / 2
  const circ = Math.PI * r // semicircle length

  // map value (0..1) to angle along top semicircle (180deg -> 0deg)
  const angle = Math.PI * (1 - value)
  const needleX = cx + r * Math.cos(angle)
  const needleY = cy - r * Math.sin(angle)

  return (
    <div className="relative" style={{ width: size, height: size / 2 + 40 }}>
      <svg width={size} height={size / 2 + 40} viewBox={`0 0 ${size} ${size / 2 + 40}`}>
        {/* coloured bands */}
        {levels.map((lvl, i) => {
          const start = i === 0 ? 0 : levels[i - 1].max
          const end = Math.min(lvl.max, 1)
          const a0 = Math.PI * (1 - start)
          const a1 = Math.PI * (1 - end)
          const x0 = cx + r * Math.cos(a0)
          const y0 = cy - r * Math.sin(a0)
          const x1 = cx + r * Math.cos(a1)
          const y1 = cy - r * Math.sin(a1)
          return (
            <path
              key={i}
              d={`M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`}
              fill="none"
              stroke={lvl.color}
              strokeWidth={stroke}
              strokeLinecap="butt"
              opacity={0.28}
            />
          )
        })}

        {/* active arc */}
        <motion.path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${needleX} ${needleY}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          initial={false}
          animate={{ d: `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${needleX} ${needleY}` }}
        />

        {/* needle hub */}
        <motion.circle
          cx={needleX}
          cy={needleY}
          r={9}
          fill="#0a0e1a"
          stroke={color}
          strokeWidth={3}
          animate={{ cx: needleX, cy: needleY }}
        />

        {/* center value */}
        <text x={cx} y={cy - 6} textAnchor="middle" className="fill-white" style={{ fontSize: 40, fontFamily: 'JetBrains Mono', fontWeight: 700 }}>
          {value.toFixed(2)}
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle" className="fill-slate-500" style={{ fontSize: 12 }}>
          P_final
        </text>
      </svg>
    </div>
  )
}
