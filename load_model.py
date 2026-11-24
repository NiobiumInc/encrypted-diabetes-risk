# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Part of a collaborative effort between hardshell.ai and Niobium.
"""
Load the diabetes XGBoost model and run inference — Niobium x Hardshell demo.

Scores the held-out TEST split (cdc_diabetes_test.parquet); the ROC-AUC printed
here matches the test AUC recorded in model/diabetes_model_metrics.json.

The model is saved in XGBoost's portable native JSON format (no pickle, no
third-party runtime lock-in).

    pip install -r requirements.txt
    python load_model.py
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model/xgb_diabetes_model.json")
TEST_PATH = os.path.join(HERE, "data/cdc_diabetes_test.parquet")
LABEL_COL = "Diabetes_binary"
THRESHOLD = 0.64  # recommended operating point: the diabetes-class max-F1 threshold, rounded


def main():
    with open(os.path.join(HERE, "model/feature_cols.json")) as fh:
        feature_cols = json.load(fh)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    df = pd.read_parquet(TEST_PATH)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.int64)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    print(f"Loaded model : {os.path.basename(MODEL_PATH)}")
    print(f"Test records : {len(df):,}  |  features: {len(feature_cols)}")
    print(f"Held-out ROC-AUC : {roc_auc_score(y, proba):.4f}   (matches diabetes_model_metrics.json)")
    print(f"Accuracy @ {THRESHOLD:.2f} : {(pred == y).mean():.4f}")
    print("\nFirst 5 predictions:")
    for i in range(5):
        print(f"  record {i}: p(diabetes)={proba[i]:.3f}  ->  {'diabetes' if pred[i] else 'no-diabetes':11s}  (actual={int(y[i])})")


if __name__ == "__main__":
    main()
