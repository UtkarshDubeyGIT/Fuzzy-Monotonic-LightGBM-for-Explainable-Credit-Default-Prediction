from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def _resolve_repo_paths() -> Tuple[Path, Path, Path]:
    """Return (repo_root, data_processed_dir, results_dir).

    Prefers repo_root/data/processed; falls back to src/data/processed if needed.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent

    primary = repo_root / 'data' / 'processed'
    fallback = repo_root / 'src' / 'data' / 'processed'

    data_dir = primary if primary.exists() else fallback
    results_dir = repo_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, data_dir, results_dir


def _load_splits(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed splits not found at {data_dir}. Expected train.csv and test.csv"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def _split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if 'default' not in df.columns:
        raise ValueError("Target column 'default' missing in processed dataset.")
    X = df.drop(columns=['default'])
    y = df['default'].astype(int)
    return X, y


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    # Required metrics: ROC-AUC, PR-AUC, Brier score, KS statistic
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier': float(brier),
        'ks': ks,
    }


def _plot_pr_curves(results: Dict[str, np.ndarray], y_true: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, probs in results.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        plt.plot(recall, precision, lw=2, label=name)
    base_rate = y_true.mean()
    plt.hlines(base_rate, 0, 1, colors='gray', linestyles='--', label=f'Baseline (p={base_rate:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Baselines)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_calibration_curves(results: Dict[str, np.ndarray], y_true: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, probs in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy='uniform')
        plt.plot(mean_pred, frac_pos, marker='o', lw=2, label=name)
    # Perfectly calibrated reference
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves (Baselines)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_baseline_models() -> Tuple[object, Dict[str, float]]:
    """Train baseline models on the preprocessed Taiwan dataset and return the best.

    Implements the required steps:
    1) Load processed splits (data/processed/train.csv, test.csv)
    2) Separate X and y (target='default')
    3) Train LogisticRegression(class_weight='balanced') and LightGBM(class_weight='balanced')
    4) Evaluate both on test: ROC-AUC, PR-AUC, Brier, KS
    5) Plot+save PR curve and calibration curve under results/
    6) Save best model's metrics as results/metrics_baseline.json (best by PR-AUC)
    7) Return (best_model, best_metrics_dict)

    Notes:
    - No accuracy metric is used.
    - No fuzzy features or monotonic constraints here.
    - This module does not modify preprocessing; it assumes prior processing.
    """

    repo_root, data_dir, results_dir = _resolve_repo_paths()
    train_df, test_df = _load_splits(data_dir)
    X_train, y_train = _split_X_y(train_df)
    X_test, y_test = _split_X_y(test_df)
    # ensure engineered features from preprocess are part of final model feature set
    engineered_cols = ["BILL_AMT_AVG","utilization","repay_ratio1","delinquency_intensity","paytrend"]

    for col in engineered_cols:
        if col not in X_train.columns:
            raise ValueError(f"Engineered feature missing: {col}")

    # Model 1: Logistic Regression (balanced)
    log_reg = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=42,
    )
    log_reg.fit(X_train, y_train)
    log_proba = log_reg.predict_proba(X_test)[:, 1]
    log_metrics = _compute_metrics(y_test.values, log_proba)

    # Model 2: LightGBM Classifier (balanced)
    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except Exception as e:  # pragma: no cover - installation environment dependent
        raise ImportError(
            "lightgbm is required for baseline training. Install with: pip install lightgbm"
        ) from e

    lgbm = LGBMClassifier(
        class_weight='balanced',
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    lgbm_metrics = _compute_metrics(y_test.values, lgbm_proba)

    # Plot curves for both
    pr_path = results_dir / 'pr_curve.png'
    _plot_pr_curves({'LogReg': log_proba, 'LightGBM': lgbm_proba}, y_test.values, pr_path)

    calib_path = results_dir / 'calibration.png'
    _plot_calibration_curves({'LogReg': log_proba, 'LightGBM': lgbm_proba}, y_test.values, calib_path)

    # Select best by PR-AUC; tie-breaker by ROC-AUC
    metric_key = 'pr_auc'
    if lgbm_metrics[metric_key] > log_metrics[metric_key] or (
        np.isclose(lgbm_metrics[metric_key], log_metrics[metric_key])
        and lgbm_metrics['roc_auc'] >= log_metrics['roc_auc']
    ):
        best_name = 'LightGBM'
        best_model = lgbm
        best_metrics = {**lgbm_metrics, 'model': best_name}
    else:
        best_name = 'LogReg'
        best_model = log_reg
        best_metrics = {**log_metrics, 'model': best_name}

    # Save best metrics
    metrics_path = results_dir / 'metrics_baseline.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, indent=2)

    return best_model, best_metrics


if __name__ == '__main__':
    model, metrics = train_baseline_models()
    print('Best baseline model:', metrics.get('model'))
    print(json.dumps(metrics, indent=2))
