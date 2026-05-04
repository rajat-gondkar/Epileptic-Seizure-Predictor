#!/usr/bin/env python3
"""
XGBoost Genetic Branch Model
===============================
Trains an XGBoost classifier on the 221-dim EEG feature vectors
augmented with 12-dim genetic profiles.

Input:  [N, 233]  (221 EEG features + 12 genetic features)
Output: P_seizure ∈ [0,1]
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report


def _detect_xgb_device():
    """Use CUDA for XGBoost if available, else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def build_xgboost_model(config):
    """Build an XGBoost classifier from config."""
    xgb_cfg = config.get("xgboost", {})
    device = _detect_xgb_device()
    return xgb.XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 300),
        max_depth=xgb_cfg.get("max_depth", 4),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=xgb_cfg.get("scale_pos_weight", 5),
        eval_metric=xgb_cfg.get("eval_metric", "auc"),
        tree_method="hist",
        device=device,
        random_state=42,
    )


def train_xgboost(X_train, y_train, X_val, y_val, config):
    """
    Train XGBoost with early stopping.

    Args:
        X_train: [N_train, D] feature array
        y_train: [N_train] labels
        X_val:   [N_val, D] feature array
        y_val:   [N_val] labels
        config:  config dict

    Returns:
        trained model, dict of metrics
    """
    model = build_xgboost_model(config)
    xgb_cfg = config.get("xgboost", {})
    early_stop = xgb_cfg.get("early_stopping_rounds", 20)

    # Robust early-stopping: try modern API, then legacy, then none.
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
        "verbose": 50,
    }

    try:
        # XGBoost >= 2.0 style
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stop, save_best=True)]
        fit_kwargs["callbacks"] = callbacks
        model.fit(**fit_kwargs)
        print(f"  Early stopping via callbacks (rounds={early_stop})")
    except TypeError as e:
        if "callbacks" not in str(e).lower():
            raise
        fit_kwargs.pop("callbacks", None)
        try:
            # XGBoost 1.x style
            fit_kwargs["early_stopping_rounds"] = early_stop
            model.fit(**fit_kwargs)
            print(f"  Early stopping via early_stopping_rounds={early_stop}")
        except TypeError as e2:
            if "early_stopping_rounds" not in str(e2).lower():
                raise
            fit_kwargs.pop("early_stopping_rounds", None)
            print("  [WARNING] XGBoost does not support early stopping; training all estimators")
            model.fit(**fit_kwargs)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "auc": float(auc),
        "accuracy": float(report["accuracy"]),
        "precision_preictal": float(report.get("1", {}).get("precision", 0)),
        "recall_preictal": float(report.get("1", {}).get("recall", 0)),
        "f1_preictal": float(report.get("1", {}).get("f1-score", 0)),
    }

    return model, metrics
