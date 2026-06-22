export function formatValue(value, format) {
  switch (format) {
    case 'decimal3':
      return value.toFixed(3)
    case 'decimal2':
      return value.toFixed(2)
    case 'percent1':
      return `+${value.toFixed(1)}%`
    case 'percent2':
      return `${value.toFixed(2)}%`
    default:
      return String(value)
  }
}

export const accentMap = {
  eeg: {
    text: 'text-eeg',
    border: 'border-eeg/30',
    bg: 'bg-eeg/10',
    ring: 'ring-eeg/40',
    glow: 'shadow-glow',
    hex: '#22d3ee',
  },
  gene: {
    text: 'text-gene',
    border: 'border-gene/30',
    bg: 'bg-gene/10',
    ring: 'ring-gene/40',
    glow: 'shadow-glow-violet',
    hex: '#a78bfa',
  },
  fuse: {
    text: 'text-fuse',
    border: 'border-fuse/30',
    bg: 'bg-fuse/10',
    ring: 'ring-fuse/40',
    glow: 'shadow-glow-emerald',
    hex: '#34d399',
  },
}

export function alertForScore(score, levels) {
  for (const lvl of levels) {
    if (score < lvl.max) return lvl
  }
  return levels[levels.length - 1]
}
