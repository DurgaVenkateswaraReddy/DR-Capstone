# train.py
# Exact methodology as your script (no changes to modeling).
# Adds persistence of scaler, all models, and ensemble threshold for use in the Flask app.

import os
import re
import json
import time
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,  # trapezoidal PR-AUC as in your code
    confusion_matrix,
)

from imblearn.combine import SMOTETomek

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

from joblib import dump

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Features exactly as in your code
image_features = ['exudates_count', 'hemorrhages_count', 'microaneurysms_count', 'vessel_tortuosity', 'macular_thickness']
clinical_features = ['fasting_glucose', 'hba1c', 'diabetes_duration']
features = image_features + clinical_features
target_col = "retinal_disorder"

def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return s if s else "model"

def evaluate_model(model, X_train, y_train, X_val, y_val, name):
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.45, 0.65, 0.05)  # 0.45, 0.50, 0.55, 0.60
    best_bal_acc, best_threshold = 0.0, 0.5
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = float(thresh)

    y_pred_final = (y_probs >= best_threshold).astype(int)

    print(f"\n== {name} ==")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Balanced Accuracy: {best_bal_acc:.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_val, y_probs):.4f}")
    precision, recall, _ = precision_recall_curve(y_val, y_probs)
    prauc = auc(recall, precision)
    print(f"PR-AUC: {prauc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_final))
    print("Classification Report:")
    print(classification_report(y_val, y_pred_final, zero_division=0))

    metrics = {
        "best_threshold": best_threshold,
        "balanced_accuracy": float(best_bal_acc),
        "auc_roc": float(roc_auc_score(y_val, y_probs)),
        "pr_auc": float(prauc),
        "confusion_matrix": confusion_matrix(y_val, y_pred_final).tolist(),
        "classification_report": classification_report(y_val, y_pred_final, zero_division=0, output_dict=True),
    }
    return y_probs, metrics

def main(data_path: str, test_size: float = 0.2):
    # Load and preprocess EXACTLY like your script
    data = pd.read_csv(data_path)

    X = data[features]
    y = data[target_col]
    mask = y.isin([0, 1])
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int).values

    # Clinical scaling first
    X.loc[:, clinical_features] = X.loc[:, clinical_features].astype(float) * 0.1

    # SMOTETomek BEFORE splitting (as in your code)
    smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    # StandardScaler fit on resampled X (as in your code)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=features)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_resampled, stratify=y_resampled, test_size=test_size, random_state=RANDOM_STATE
    )

    # Models dict with same params and scale_pos_weight/class_weight where applicable
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)  # avoid div by zero
    models = {
        "XGBoost": xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,  # keep to match your code
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=spw,
            n_jobs=-1,
            tree_method="hist",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            solver='liblinear', class_weight='balanced',
            max_iter=500, random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=10,
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "CatBoost": CatBoostClassifier(
            iterations=100, depth=8, learning_rate=0.05,
            scale_pos_weight=spw, verbose=0, random_seed=RANDOM_STATE,
            loss_function="Logloss",
        ),
    }

    # Train/evaluate each model
    probs = {}
    per_model_metrics = {}
    fitted_models = {}
    for name, model in models.items():
        y_probs, m = evaluate_model(model, X_train, y_train, X_val, y_val, name)
        probs[name] = y_probs
        per_model_metrics[name] = m
        fitted_models[name] = model  # already fit inside evaluate_model

    # Ensemble (mean of model probabilities)
    ensemble_probs = np.mean(np.array(list(probs.values())), axis=0)
    thresholds = np.arange(0.45, 0.65, 0.05)
    best_bal_acc, best_thresh = 0.0, 0.5
    for thr in thresholds:
        y_pred = (ensemble_probs >= thr).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = float(thr)
    y_pred_final = (ensemble_probs >= best_thresh).astype(int)

    print("\nEnsemble results")
    print(f"Threshold: {best_thresh:.2f}")
    print(f"Balanced Accuracy: {best_bal_acc:.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_val, ensemble_probs):.4f}")
    precision, recall, _ = precision_recall_curve(y_val, ensemble_probs)
    prauc = auc(recall, precision)
    print(f"PR-AUC: {prauc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_final))
    print("Classification Report:")
    print(classification_report(y_val, y_pred_final, zero_division=0))

    ensemble_metrics = {
        "threshold": best_thresh,
        "balanced_accuracy": float(best_bal_acc),
        "auc_roc": float(roc_auc_score(y_val, ensemble_probs)),
        "pr_auc": float(prauc),
        "confusion_matrix": confusion_matrix(y_val, y_pred_final).tolist(),
        "classification_report": classification_report(y_val, y_pred_final, zero_division=0, output_dict=True),
    }

    # Persist artifacts for the web app inference (ensemble)
    # 1) Scaler
    dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    # 2) Models
    model_filenames = {}
    for name, mdl in fitted_models.items():
        fname = f"{safe_name(name)}.pkl"
        dump(mdl, os.path.join(MODEL_DIR, fname))
        model_filenames[name] = fname

    # 3) Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "image_features": image_features,
        "clinical_features": clinical_features,
        "clinical_scale_factor": 0.1,
        "threshold_grid": list(map(float, thresholds)),
        "ensemble_threshold": float(best_thresh),
        "model_order": list(models.keys()),
        "model_filenames": model_filenames,
        "random_state": RANDOM_STATE,
        "data_path": os.path.abspath(data_path),
    }
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 4) Metrics
    metrics_out = {
        "per_model": per_model_metrics,
        "ensemble": ensemble_metrics
    }
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nSaved:")
    print(f"- Scaler: {os.path.join(ARTIFACT_DIR, 'scaler.pkl')}")
    print(f"- Models: {MODEL_DIR}/*.pkl")
    print(f"- Metadata: {os.path.join(ARTIFACT_DIR, 'metadata.json')}")
    print(f"- Metrics: {os.path.join(ARTIFACT_DIR, 'metrics.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to your CSV (e.g., dr 8.csv)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation size (default 0.2)")
    args = parser.parse_args()
    main(args.data, test_size=args.test_size)