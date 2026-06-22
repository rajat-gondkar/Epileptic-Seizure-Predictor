export const API_BASE =
  import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export async function checkHealth(timeoutMs = 2500) {
  const ctrl = new AbortController()
  const id = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: ctrl.signal })
    clearTimeout(id)
    if (!res.ok) return { online: false }
    const data = await res.json()
    return { online: data.status === 'ok' && data.model_loaded, info: data }
  } catch {
    clearTimeout(id)
    return { online: false }
  }
}

export async function predictEdf(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: form })
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const e = await res.json()
      if (e.detail) detail = e.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

/**
 * Build a realistic SIMULATED inference result for offline/no-EDF demos.
 * Clearly labelled in the UI as simulated — not real model output.
 */
export function buildSimulatedResult() {
  const nWindows = 80
  const windows = []
  // scripted trajectory: long interictal -> preictal ramp -> ictal burst -> recovery
  const preictalStart = 48
  const ictalStart = 64
  const ictalEnd = 69

  for (let i = 0; i < nWindows; i++) {
    let pPre, pIct, pInt
    if (i < preictalStart) {
      pPre = 0.04 + Math.random() * 0.06
      pIct = 0.005 + Math.random() * 0.01
    } else if (i < ictalStart) {
      const r = (i - preictalStart) / (ictalStart - preictalStart)
      pPre = 0.2 + r * 0.6 + (Math.random() - 0.5) * 0.08
      pIct = 0.02 + r * 0.08
    } else if (i <= ictalEnd) {
      pPre = 0.25 + (Math.random() - 0.5) * 0.1
      pIct = 0.6 + Math.random() * 0.25
    } else {
      const r = Math.min((i - ictalEnd) / 6, 1)
      pPre = 0.4 * (1 - r) + 0.05
      pIct = 0.2 * (1 - r) + 0.01
    }
    pPre = Math.max(0, Math.min(1, pPre))
    pIct = Math.max(0, Math.min(1, pIct))
    pInt = Math.max(0, 1 - pPre - pIct)
    const total = pInt + pPre + pIct
    pInt /= total
    pPre /= total
    pIct /= total
    const probs = [pInt, pPre, pIct]
    const pred = probs.indexOf(Math.max(...probs))

    // synthetic preview trace: amplitude/sharpness rises during ictal
    const amp = pred === 2 ? 1 : pred === 1 ? 0.6 : 0.4
    const preview = Array.from({ length: 70 }, (_, k) => {
      const t = k / 70
      let v = Math.sin(t * 40 + i) * 0.5 + Math.sin(t * 13 + i * 2) * 0.3
      if (pred === 2) v += Math.sin(t * 90) * 0.5 * Math.sin(t * 6)
      v += (Math.random() - 0.5) * 0.2
      return +(v * amp).toFixed(3)
    })

    windows.push({
      index: i,
      start_sec: i * 30,
      end_sec: (i + 1) * 30,
      pred_class: pred,
      pred_name: ['interictal', 'preictal', 'ictal'][pred],
      p_interictal: +pInt.toFixed(4),
      p_preictal: +pPre.toFixed(4),
      p_ictal: +pIct.toFixed(4),
      preview,
    })
  }

  const counts = { interictal: 0, preictal: 0, ictal: 0 }
  windows.forEach((w) => (counts[w.pred_name] += 1))

  return {
    filename: 'sample_chb15_recording.edf (simulated)',
    sampling_freq: 256,
    duration_sec: nWindows * 30,
    channels_present: 19,
    channels_expected: 19,
    n_windows: nWindows,
    windows,
    summary: {
      counts,
      any_preictal: counts.preictal > 0,
      any_ictal: counts.ictal > 0,
      max_preictal_prob: Math.max(...windows.map((w) => w.p_preictal)),
      max_ictal_prob: Math.max(...windows.map((w) => w.p_ictal)),
    },
    simulated: true,
  }
}
