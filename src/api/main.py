#!/usr/bin/env python3
"""
FastAPI service exposing real BiLSTM seizure-prediction inference.

Run from the project root:
    ./venv/bin/uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /api/health      -> service + model status
    GET  /api/model-info  -> architecture metadata
    POST /api/predict     -> upload an .edf file, get per-window predictions
"""

import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import SeizureInferenceEngine

app = FastAPI(title="EEG-Genetic Fusion · Seizure Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo: allow the Vite dev server (localhost:5173) etc.
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy single-instance engine so model loads once.
_engine = None


def get_engine() -> SeizureInferenceEngine:
    global _engine
    if _engine is None:
        _engine = SeizureInferenceEngine()
    return _engine


@app.get("/api/health")
def health():
    try:
        eng = get_engine()
        return {"status": "ok", "model_loaded": True, "model_file": eng.meta["model_file"]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "model_loaded": False, "detail": str(exc)}


@app.get("/api/model-info")
def model_info():
    return get_engine().meta


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), max_windows: int = 240):
    name = (file.filename or "").lower()
    if not name.endswith(".edf"):
        raise HTTPException(status_code=400, detail="Please upload a .edf file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = get_engine().predict_edf(tmp_path, max_windows=max_windows)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/")
def root():
    return {
        "service": "EEG-Genetic Fusion Seizure Inference API",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/model-info", "/api/predict"],
    }
