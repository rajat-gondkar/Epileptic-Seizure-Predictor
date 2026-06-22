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

## Live inference demo
The **Live Demo** section runs the real trained BiLSTM on an uploaded CHB-MIT
`.edf` file via a FastAPI backend, classifying every 30-second window as
interictal / preictal / ictal.

Start the backend from the project root:

```bash
./venv/bin/python -m uvicorn src.api.main:app --port 8000
# or: bash scripts/run_inference_api.sh
```

The dashboard auto-detects the backend at `http://localhost:8000`. If it's
offline (or you have no `.edf` handy), use **"Play sample (simulated)"** — a
clearly-labelled illustrative playback so the demo always works.

To point the dashboard at a different backend URL, set `VITE_API_BASE`:

```bash
VITE_API_BASE=http://192.168.1.50:8000 npm run dev
```

CHB-MIT `.edf` files can be downloaded from
[physionet.org/content/chbmit](https://physionet.org/content/chbmit/1.0.0/).
