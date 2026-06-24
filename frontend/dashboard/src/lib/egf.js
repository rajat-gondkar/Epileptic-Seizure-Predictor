// Parser for the proprietary .egf ("EEG-Genetic Fusion bundle") demo format.
// Decodes a patient bundle and runs the documented fusion blend per window:
//     P_final = alpha * P_eeg + (1 - alpha) * P_genetic

const FEATURE_NAMES = [
  'SCN1A_mutation', 'SCN8A_mutation', 'KCNQ2_mutation', 'SCN2A_mutation',
  'KCNT1_mutation', 'DEPDC5_mutation', 'PCDH19_mutation', 'GRIN2A_mutation',
  'GABRA1_mutation', 'SCN1A_pLI', 'SCN8A_pLI', 'KCNQ2_pLI', 'SCN2A_pLI',
  'GRIN2A_pLI', 'PCDH19_pLI', 'GABRA1_pLI', 'polygenic_risk_score',
  'sodium_channel_interaction', 'potassium_channel_interaction',
  'receptor_interaction', 'prs_tier1_interaction', 'mutation_burden',
]

function b64decode(s) {
  // browser-safe base64 -> utf8 string
  if (typeof atob === 'function') return decodeURIComponent(escape(atob(s)))
  return Buffer.from(s, 'base64').toString('utf8') // node fallback (tests)
}

function clamp(x, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, x))
}

export function parseEgf(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
  if (!lines.length || !lines[0].startsWith('EGFUSION')) {
    throw new Error('Not a valid .egf fusion bundle (missing EGFUSION header).')
  }

  const metaLine = lines.find((l) => l.startsWith('META|'))
  const geneticLine = lines.find((l) => l.startsWith('GENETIC|'))
  const payloadLine = lines.find((l) => l.startsWith('PAYLOAD|'))
  if (!metaLine || !payloadLine) {
    throw new Error('Corrupt .egf bundle: missing META or PAYLOAD section.')
  }

  // --- META ---
  const meta = {}
  metaLine
    .slice('META|'.length)
    .split('|')
    .forEach((kv) => {
      const [k, v] = kv.split('=')
      meta[k] = v
    })
  const alpha = parseFloat(meta.alpha ?? '0.5')
  const windowSec = parseInt(meta.window_sec ?? '30', 10)
  const patient = meta.patient ?? 'unknown'
  const genes = meta.genes && meta.genes !== 'none' ? meta.genes.split(';') : []

  // --- GENETIC vector ---
  let geneticVector = []
  if (geneticLine) {
    geneticVector = b64decode(geneticLine.slice('GENETIC|'.length))
      .split(',')
      .map((v) => parseFloat(v))
  }

  // --- PAYLOAD (per-window) ---
  const rows = b64decode(payloadLine.slice('PAYLOAD|'.length)).split('|')
  const windows = rows.map((row) => {
    const [idx, eeg, gen, phase] = row.split(',')
    const eegScore = clamp(parseFloat(eeg))
    const geneticScore = clamp(parseFloat(gen))
    const fused = clamp(alpha * eegScore + (1 - alpha) * geneticScore)
    const i = parseInt(idx, 10)
    const ph = parseInt(phase, 10)
    return {
      index: i,
      start_sec: i * windowSec,
      end_sec: (i + 1) * windowSec,
      eeg_score: eegScore,
      genetic_score: geneticScore,
      alpha,
      fused,
      phase: ph,
      preview: buildPreview(i, ph),
    }
  })

  const maxFused = Math.max(...windows.map((w) => w.fused))
  const maxEeg = Math.max(...windows.map((w) => w.eeg_score))
  const geneticScore = windows.length ? windows[0].genetic_score : 0

  return {
    patient,
    genes,
    alpha,
    genetic_vector: geneticVector,
    feature_names: FEATURE_NAMES,
    genetic_score: geneticScore,
    n_windows: windows.length,
    duration_sec: windows.length * windowSec,
    windows,
    summary: {
      max_fused: maxFused,
      max_eeg: maxEeg,
      genetic_score: geneticScore,
    },
  }
}

// Synthetic preview trace shaped by phase (the .egf carries scores, not raw EEG)
function buildPreview(seed, phase) {
  const amp = phase === 2 ? 1 : phase === 1 ? 0.6 : 0.4
  const out = []
  for (let k = 0; k < 70; k++) {
    const t = k / 70
    let v = Math.sin(t * 38 + seed) * 0.5 + Math.sin(t * 12 + seed * 2) * 0.3
    if (phase === 2) v += Math.sin(t * 88) * 0.5 * Math.sin(t * 6)
    v += (Math.sin(k * 12.9 + seed * 7.1) - 0.5) * 0.18
    out.push(+(v * amp).toFixed(3))
  }
  return out
}
