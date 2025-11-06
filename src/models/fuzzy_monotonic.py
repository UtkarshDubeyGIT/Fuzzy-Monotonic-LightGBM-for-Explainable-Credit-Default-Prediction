from __future__ import annotations

import json  # ensure available for final JSON output
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)


def _resolve_paths() -> Tuple[Path, Path, Path, Path]:
    """Return (repo_root, processed_dir, results_dir, raw_csv_path)."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    processed_dir = repo_root / 'data' / 'processed'
    if not processed_dir.exists():
        # fallback to src/data/processed
        processed_dir = repo_root / 'src' / 'data' / 'processed'
    results_dir = repo_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = repo_root / 'src' / 'data' / 'taiwan_credit.csv'
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv_path}")
    return repo_root, processed_dir, results_dir, raw_csv_path


def _load_processed(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(processed_dir / 'train.csv')
    test = pd.read_csv(processed_dir / 'test.csv')
    if 'default' not in train.columns or 'default' not in test.columns:
        raise ValueError("Processed splits must contain 'default' column.")
    return train, test


def _read_raw_df(raw_csv_path: Path) -> pd.DataFrame:
    # Handle second-row header and drop potential unnamed column
    df = pd.read_csv(raw_csv_path, header=1)
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    elif 'Y' in df.columns:
        df = df.rename(columns={'Y': 'default'})
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    df['default'] = df['default'].astype(int)
    # Create BILL_AMT_AVG on raw for fuzzy bases
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['BILL_AMT_AVG'] = df[bill_cols].mean(axis=1)
    # Engineered bases for fuzzy (raw, unscaled)
    df['utilization'] = (df['BILL_AMT_AVG'] / df['LIMIT_BAL'].replace(0, pd.NA)).fillna(0.0).clip(0, 1)
    df['repay_ratio1'] = (df['PAY_AMT1'] / df['BILL_AMT1'].replace(0, pd.NA)).fillna(0.0).clip(0, 1)
    pay_cols_ = [f'PAY_{i}' for i in range(0, 7) if f'PAY_{i}' in df.columns]
    df['delinquency_intensity'] = df[pay_cols_].max(axis=1)
    df['paytrend'] = ((df['PAY_AMT6'] - df['PAY_AMT1']) / (df['PAY_AMT1'] + 1e-6)).clip(-1, 1)

    return df


def _split_raw_like_processed(raw_df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = raw_df.drop(columns=['default'])
    y = raw_df['default']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _fit_fuzzy_cutpoints(series: pd.Series) -> Dict[str, float]:
    # Use robust percentiles to define Low/Med/High regions
    vals = series.astype(float).values
    q05 = float(np.percentile(vals, 5))
    q33 = float(np.percentile(vals, 33))
    q50 = float(np.percentile(vals, 50))
    q66 = float(np.percentile(vals, 66))
    q95 = float(np.percentile(vals, 95))
    return {
        'min': float(np.min(vals)),
        'q05': q05,
        'q33': q33,
        'q50': q50,
        'q66': q66,
        'q95': q95,
        'max': float(np.max(vals)),
    }


def _compute_memberships(values: np.ndarray, cp: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import skfuzzy as fuzz
    # Define trapezoids/triangles for Low/Med/High
    # Low: [min, min, q33, q50]
    low = fuzz.membership.trapmf(values, [cp['min'], cp['min'], cp['q33'], cp['q50']])
    # Medium: [q33, q50, q66] triangular
    med = fuzz.membership.trimf(values, [cp['q33'], cp['q50'], cp['q66']])
    # High: [q50, q66, max, max]
    high = fuzz.membership.trapmf(values, [cp['q50'], cp['q66'], cp['max'], cp['max']])
    return low.astype(float), med.astype(float), high.astype(float)


def _build_fuzzy_features(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    bases: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Return (train_memberships, test_memberships, cutpoints_by_var)."""
    cutpoints_by_var: Dict[str, Dict[str, float]] = {}
    train_feats = []
    test_feats = []
    for var in bases:
        cp = _fit_fuzzy_cutpoints(X_train_raw[var])
        cutpoints_by_var[var] = cp
        tr_low, tr_med, tr_high = _compute_memberships(X_train_raw[var].values.astype(float), cp)
        te_low, te_med, te_high = _compute_memberships(X_test_raw[var].values.astype(float), cp)
        train_feats.append(pd.DataFrame({
            f'{var}_low': tr_low,
            f'{var}_med': tr_med,
            f'{var}_high': tr_high,
        }, index=X_train_raw.index))
        test_feats.append(pd.DataFrame({
            f'{var}_low': te_low,
            f'{var}_med': te_med,
            f'{var}_high': te_high,
        }, index=X_test_raw.index))
    train_m = pd.concat(train_feats, axis=1)
    test_m = pd.concat(test_feats, axis=1)
    return train_m, test_m, cutpoints_by_var


def _min_and(*cols: List[pd.Series]) -> pd.Series:
    arrs = [c.values for c in cols]
    return pd.Series(np.min(np.vstack(arrs), axis=0), index=cols[0].index)


def _build_fuzzy_rules(train_m: pd.DataFrame, test_m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 10 illustrative rules using min-AND
    # Names for readability
    def tr(name: str) -> pd.Series: return train_m[name]
    def te(name: str) -> pd.Series: return test_m[name]

    rules = [
    # High-risk patterns
    ('fuzz_rule1',  ('utilization_high', 'repay_ratio1_low')),         # high util + low repay
    ('fuzz_rule2',  ('delinquency_intensity_high',)),                   # strong lateness
    ('fuzz_rule3',  ('PAY_0_high', 'LIMIT_BAL_low')),                   # recent late + low limit
    ('fuzz_rule4',  ('paytrend_high', 'repay_ratio1_low')),             # paying down ↓ (neg trend) + low repay
    ('fuzz_rule5',  ('AGE_low', 'PAY_0_high')),                         # young + late
    ('fuzz_rule6',  ('utilization_high', 'PAY_0_high')),                # maxed out + late

    # Low-risk patterns
    ('fuzz_rule7',  ('utilization_low', 'repay_ratio1_high')),          # low util + high repay
    ('fuzz_rule8',  ('PAY_0_low', 'repay_ratio1_high')),                # on time + high repay
    ('fuzz_rule9',  ('AGE_high', 'PAY_0_low')),                         # older + on time
    ('fuzz_rule10', ('LIMIT_BAL_high', 'PAY_0_low')),                   # high limit + on time
    ('fuzz_rule11', ('paytrend_low', 'repay_ratio1_high')),             # (pos trend) better pay + high repay
    ('fuzz_rule12', ('utilization_low',)),                              # very low utilization alone
    ]


    tr_df = pd.DataFrame(index=train_m.index)
    te_df = pd.DataFrame(index=test_m.index)
    for name, terms in rules:
        tr_cols = [tr(t) for t in terms]
        te_cols = [te(t) for t in terms]
        tr_df[name] = _min_and(*tr_cols)
        te_df[name] = _min_and(*te_cols)
    return tr_df, te_df


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    roc_auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)
    brier = brier_score_loss(y_true, prob)
    fpr, tpr, _ = roc_curve(y_true, prob)
    ks = float(np.max(tpr - fpr))
    return {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc), 'brier': float(brier), 'ks': ks}


def train_fuzzy_monotonic_model(*, skip_monotonic: bool = False) -> Tuple[object, Dict[str, float]]:
    """Train fuzzy-featured, monotonic-constrained LightGBM on Taiwan dataset.

    Steps:
    1) Load processed splits and raw copies for fuzzy calc
    2) Create BILL_AMT_AVG on raw
    3) Bases: LIMIT_BAL, BILL_AMT_AVG, AGE, PAY_0, PAY_AMT1
    4) Fit Low/Med/High cut points on TRAIN only
    5) Build 10 fuzzy rules via min-AND
    6) Append fuzzy memberships + rules to processed features
    7) Train LGBM with class_weight='balanced' and specified monotonic constraints
    8) Evaluate and save metrics to results/metrics_fuzzy.json
    9) Save SHAP summary to results/shap_fuzzy.png
    10) Return (model, metrics)
    """

    # Load splits
    repo_root, processed_dir, results_dir, raw_csv_path = _resolve_paths()
    proc_train, proc_test = _load_processed(processed_dir)
    X_train_proc = proc_train.drop(columns=['default'])
    y_train = proc_train['default'].astype(int)
    X_test_proc = proc_test.drop(columns=['default'])
    y_test = proc_test['default'].astype(int)

    # Load raw and split with same settings for fuzzy-only calculations
    raw_df = _read_raw_df(raw_csv_path)
    X_train_raw, X_test_raw, _, _ = _split_raw_like_processed(raw_df)

    # Ensure BILL_AMT_AVG exists in processed features as a raw (unscaled) column
    # We compute from raw and append to processed matrices (tree model tolerates scale mix)
    # bill_avg_train = X_train_raw['BILL_AMT_AVG'].reset_index(drop=True)
    # bill_avg_test = X_test_raw['BILL_AMT_AVG'].reset_index(drop=True)
    # X_train_proc = X_train_proc.reset_index(drop=True)
    # X_test_proc = X_test_proc.reset_index(drop=True)
    # X_train_proc['BILL_AMT_AVG'] = bill_avg_train.values
    # X_test_proc['BILL_AMT_AVG'] = bill_avg_test.values

    # Build fuzzy memberships from train-only cutpoints on selected bases
    bases = [
    'utilization',          # ↑ → more risk
    'repay_ratio1',         # ↑ → less risk
    'delinquency_intensity',# ↑ → more risk
    'PAY_0',                # ↑ (later) → more risk
    'LIMIT_BAL',            # ↑ → less risk
    'AGE',                  # ↑ → less risk (mild)
    'paytrend'              # ↑ → less risk (this MUST be included)
    ]

    train_m, test_m, _ = _build_fuzzy_features(X_train_raw[bases], X_test_raw[bases], bases)

    # Rules
    tr_rules, te_rules = _build_fuzzy_rules(train_m, test_m)

    # Append fuzzy features to processed matrices
    X_train = pd.concat([X_train_proc, train_m.reset_index(drop=True), tr_rules.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_proc, test_m.reset_index(drop=True), te_rules.reset_index(drop=True)], axis=1)

    # Prepare monotonic constraints vector aligned with column order
    monotone_map = {
    'LIMIT_BAL': -1,
    'AGE': -1,
    'PAY_0': +1,
    # engineered (already in processed X)
    'utilization': +1,
    'repay_ratio1': -1,
    'delinquency_intensity': +1,
    'paytrend': -1,  # improving payments (higher) → less risk
    # keep 0 for all fuzzy_* columns
    }

    constraints: List[int] = []
    if not skip_monotonic:
        for col in X_train.columns:
            if col in monotone_map:
                constraints.append(monotone_map[col])
            elif col.startswith('fuzz_rule') or col.endswith('_low') or col.endswith('_med') or col.endswith('_high'):
                constraints.append(0)
            else:
                constraints.append(0)

    # Train LightGBM with monotonic constraints
    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("lightgbm is required. Install with: pip install lightgbm") from e

    model = LGBMClassifier(
        class_weight='balanced',
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        monotone_constraints=(None if skip_monotonic else constraints),
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = _metrics(y_test.values, y_prob)

    # Persist metrics
    with open(results_dir / 'metrics_fuzzy.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # SHAP summary plot
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        # SHAP values for test subset for speed
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        # For binary, shap_values is list [class0, class1]; take class1
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1]
        else:
            sv = shap_values
        shap.summary_plot(sv, X_test, show=False, max_display=25)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_fuzzy.png', dpi=150)
        plt.close()
    except Exception:
        # SHAP is optional; skip if not available
        pass

    return model, metrics


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train fuzzy (monotonic) model")
    parser.add_argument("--metrics-out", type=str, required=False, default=None,
                        help="Optional path to write metrics JSON")
    parser.add_argument("--variant", type=str, required=False, default=None)
    parser.add_argument("--skip-monotonic", action="store_true", default=False)
    args = parser.parse_args()

    metrics_out_path = Path(args.metrics_out).resolve() if args.metrics_out else None
    if metrics_out_path is not None:
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Train and get metrics
    _, metrics = train_fuzzy_monotonic_model(skip_monotonic=bool(args.skip_monotonic))

    # Save metrics if requested
    if metrics_out_path is not None:
        with open(metrics_out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    # Print exactly one JSON
    print(json.dumps(metrics, indent=2))
