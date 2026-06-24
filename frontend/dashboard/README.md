# NeuroGenix — EEG–Genetic Fusion Showcase Dashboard

A reactive single-page dashboard that walks through the entire seizure-prediction
pipeline: datasets → EEG preprocessing → genetic feature engineering → CTGAN →
model training → attention-gated fusion → results, with an interactive **fusion
simulator** at its centre.

## Stack
- React 18 + Vite
- Tailwind CSS (custom dark "lab" theme)
- Recharts (charts) + Framer Motion (animations)
- All data is static JSON in `src/data/`, extracted from the project's real results
  (`fusion_evaluation.json`, `fusion_metrics.json`, `xgboost_genetic_metrics.json`,
  and the technical reference). No backend required.

## Run

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build into dist/
npm run preview  # preview the production build
```

## Editing the numbers
Everything displayed lives in `src/data/*.json`:

| File | Drives |
|------|--------|
| `project.json` | hero, headline stats, team |
| `datasets.json` | dataset cards, per-patient & per-gene charts |
| `preprocessing.json` | pipeline steps, class distribution, 3-zone labels |
| `genetic.json` | 22-dim feature vector, gene tiers |
| `ctgan.json` | KS validation results |
| `training.json` | EEG loss curve, confusion matrix, XGBoost metrics |
| `fusion.json` | EEG-vs-fusion comparison, alert thresholds, simulator config |

## Fusion simulator
The simulator uses the documented blend `P_final = α · P_eeg + (1 − α) · P_genetic`
and maps the result to the 4-level clinical alert system. α defaults to the
model's learned value (≈0.50).

## Live fusion demo
The **Live Fusion Inference** section runs the attention-gated fusion model on a
patient bundle and animates the per-window result: EEG-branch score, genetic-branch
score, the attention gate α, the fused risk `P_final`, and the 4-level clinical alert.

Input is a proprietary `.egf` ("EEG-Genetic Fusion bundle") file — a base64-encoded
container holding a patient's 22-dim genetic vector plus a per-window sequence of
EEG/genetic scores. It runs entirely in the browser, so **no backend is required**.

Sample bundles live in `public/samples/` and can be loaded with one click, or you
can upload your own `.egf`. Regenerate the samples with:

```bash
./venv/bin/python scripts/generate_fusion_demo_samples.py
```

The three bundled scenarios:
- `EGF-7731` — focal seizure with a clear preictal ramp then ictal burst
- `EGF-4490` — high genetic risk (SCN1A/SCN2A) where fusion crosses alert thresholds earlier
- `EGF-2208` — stable recording, stays low-risk (true negative)

> A separate FastAPI backend (`src/api/`) that runs the raw BiLSTM on real `.edf`
> files also exists in the repo, but the dashboard demo uses the self-contained
> `.egf` fusion path so it always works offline.
