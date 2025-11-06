# API Reference

## Overview

This document provides detailed API documentation for all public functions and classes in the codebase. For usage examples, see [USAGE.md](USAGE.md).

---

## Data Module: `src/data/preprocess.py`

### `main()`

Entry point for data preprocessing pipeline.

**Signature**:

```python
def main() -> None
```

**Parameters**: None (uses CLI arguments)

**CLI Arguments**:

- `--engineered` (flag): Generate engineered features

**Behavior**:

1. Parses command-line arguments
2. Loads Taiwan credit dataset from `Data/taiwan_default_of_credit_card_clients.csv`
3. Optionally applies feature engineering
4. Splits data (80/20 stratified)
5. Scales numeric features
6. Saves train/test to `data/processed_*/`

**Outputs**:

- `data/processed_baseline_raw/train.csv` (24,000 × 23)
- `data/processed_baseline_raw/test.csv` (6,000 × 23)
- `data/processed_baseline_engineered/train.csv` (24,000 × 28)
- `data/processed_baseline_engineered/test.csv` (6,000 × 28)

**Example**:

```bash
python src/data/preprocess.py --engineered
```

**Raises**:

- `FileNotFoundError`: If raw CSV not found
- `ValueError`: If dataset missing required columns

---

### `_read_taiwan_csv(data_path: Path) -> pd.DataFrame`

Reads and cleans the Taiwan credit dataset.

**Parameters**:

- `data_path` (Path): Absolute path to raw CSV file

**Returns**:

- `pd.DataFrame`: Cleaned dataset with standardized column names

**Behavior**:

1. Reads CSV with Excel dialect
2. Removes unnecessary columns (ID, metadata)
3. Renames columns (e.g., `default.payment.next.month` → `default`)
4. Validates data types

**Example**:

```python
from pathlib import Path
df = _read_taiwan_csv(Path('Data/taiwan_default_of_credit_card_clients.csv'))
print(df.shape)  # (30000, 24)
```

---

### `_add_engineered_features(df: pd.DataFrame) -> pd.DataFrame`

Generates 5 domain-expert features.

**Parameters**:

- `df` (pd.DataFrame): Raw dataset with original 23 features

**Returns**:

- `pd.DataFrame`: Dataset with 28 features (23 + 5 engineered)

**Engineered Features**:

1. **BILL_AMT_AVG**: Mean of BILL_AMT1-6

   ```python
   BILL_AMT_AVG = (BILL_AMT1 + BILL_AMT2 + ... + BILL_AMT6) / 6
   ```

2. **utilization**: Credit utilization ratio

   ```python
   utilization = BILL_AMT1 / LIMIT_BAL
   ```

3. **repay_ratio1**: Recent repayment ratio

   ```python
   repay_ratio1 = PAY_AMT1 / (BILL_AMT1 + 1e-6)
   ```

4. **delinquency_intensity**: Cumulative payment delay

   ```python
   delinquency_intensity = sum([PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6])
   ```

5. **paytrend**: Payment behavior trend
   ```python
   paytrend = PAY_0 - PAY_6
   ```

**Example**:

```python
df_raw = pd.read_csv('Data/taiwan_default_of_credit_card_clients.csv')
df_eng = _add_engineered_features(df_raw)
print(df_eng.columns[-5:])
# ['BILL_AMT_AVG', 'utilization', 'repay_ratio1',
#  'delinquency_intensity', 'paytrend']
```

**Raises**:

- `KeyError`: If required input features missing

---

### `_split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]`

Stratified train/test split.

**Parameters**:

- `df` (pd.DataFrame): Full dataset with `default` column

**Returns**:

- `Tuple[X_train, X_test, y_train, y_test]`

**Split Strategy**:

- 80% train, 20% test
- Stratified by `default` (preserves class distribution)
- Random state = 42 (reproducibility)

**Example**:

```python
X_train, X_test, y_train, y_test = _split_data(df)
print(y_train.value_counts(normalize=True))
# 0    0.778
# 1    0.222  (same distribution as full dataset)
```

---

### `_apply_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`

Scales numeric features using RobustScaler.

**Parameters**:

- `X_train` (pd.DataFrame): Training features
- `X_test` (pd.DataFrame): Test features

**Returns**:

- `Tuple[X_train_scaled, X_test_scaled]`: Scaled DataFrames

**Behavior**:

1. Identifies numeric columns (excludes categorical like SEX, EDUCATION)
2. Fits RobustScaler on training set
3. Transforms both train and test
4. Returns DataFrames with same structure

**Scaling Method**: RobustScaler

- Centers data using median
- Scales using IQR (25th-75th percentile range)
- Robust to outliers

**Example**:

```python
X_train_scaled, X_test_scaled = _apply_scaling(X_train, X_test)
# Check scaling
print(X_train_scaled['LIMIT_BAL'].median())  # ~0.0
print(X_train_scaled['LIMIT_BAL'].quantile([0.25, 0.75]))
# 25%   -1.0
# 75%    1.0
```

---

## Baseline Models: `src/models/baseline.py`

### `train_baseline_models() -> Tuple[object, Dict[str, float]]`

Trains and compares Logistic Regression vs LightGBM.

**Parameters**: None (loads data from disk)

**Returns**:

- `Tuple[best_model, best_metrics]`
  - `best_model`: Trained scikit-learn model (LogisticRegression or LGBMClassifier)
  - `best_metrics` (Dict): Performance metrics for best model

**Process**:

1. Load `data/processed_baseline_raw/train.csv` and `test.csv`
2. Train Logistic Regression (balanced)
3. Compute metrics (ROC-AUC, PR-AUC, Brier, KS)
4. Train LightGBM (balanced)
5. Compute metrics
6. Select best model (highest PR-AUC)
7. Plot PR curves and calibration curves
8. Save metrics to `results/metrics_baseline.json`

**Hyperparameters**:

**Logistic Regression**:

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
```

**LightGBM**:

```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)
```

**Example**:

```python
model, metrics = train_baseline_models()
print(metrics)
# {
#   'roc_auc': 0.7496,
#   'pr_auc': 0.4872,
#   'brier': 0.1725,
#   'ks': 0.3821
# }

# Make predictions
import pandas as pd
X_test = pd.read_csv('data/processed_baseline_raw/test.csv')
y_test = X_test.pop('default')
probs = model.predict_proba(X_test)[:, 1]
```

**Outputs**:

- `results/metrics_baseline.json`
- `results/pr_curve.png`
- `results/calibration.png`

**Raises**:

- `FileNotFoundError`: If preprocessed data missing

---

### `_compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]`

Computes comprehensive classification metrics.

**Parameters**:

- `y_true` (np.ndarray): Ground truth labels (0 or 1)
- `y_prob` (np.ndarray): Predicted probabilities (0.0 to 1.0)

**Returns**:

- `Dict[str, float]`: Metrics dictionary

**Metrics Computed**:

1. **ROC-AUC**: Area under ROC curve
2. **PR-AUC**: Area under Precision-Recall curve (critical for imbalanced data)
3. **Brier Score**: Mean squared error of probabilities (lower is better)
4. **KS Statistic**: Kolmogorov-Smirnov statistic (max separation between classes)

**Example**:

```python
import numpy as np
y_true = np.array([0, 1, 0, 1, 0])
y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.1])

metrics = _compute_metrics(y_true, y_prob)
print(metrics)
# {
#   'roc_auc': 1.0,
#   'pr_auc': 1.0,
#   'brier': 0.05,
#   'ks': 1.0
# }
```

**Interpretation**:

- **ROC-AUC** ∈ [0, 1]: 0.5 = random, 1.0 = perfect
- **PR-AUC** ∈ [0, 1]: Baseline = prevalence (e.g., 0.22 for 22% default rate)
- **Brier** ∈ [0, 1]: 0 = perfect calibration, 0.25 = random guessing
- **KS** ∈ [0, 1]: 0 = no separation, 1.0 = perfect separation

---

### `_plot_pr_curves(results: List[Tuple], y_true: np.ndarray, out_path: Path) -> None`

Generates Precision-Recall curve comparison plot.

**Parameters**:

- `results` (List[Tuple]): List of (model_name, y_prob) tuples
- `y_true` (np.ndarray): Ground truth labels
- `out_path` (Path): Output file path (e.g., `results/pr_curve.png`)

**Returns**: None (saves plot to disk)

**Example**:

```python
from pathlib import Path
results = [
    ('Logistic Regression', lr_probs),
    ('LightGBM', lgb_probs)
]
_plot_pr_curves(results, y_test, Path('results/pr_curve.png'))
```

**Plot Features**:

- Precision vs Recall curves for each model
- PR-AUC scores in legend
- Baseline (no-skill) reference line
- Grid, labels, title

---

### `_plot_calibration_curves(results: List[Tuple], y_true: np.ndarray, out_path: Path) -> None`

Generates calibration reliability diagram.

**Parameters**:

- `results` (List[Tuple]): List of (model_name, y_prob) tuples
- `y_true` (np.ndarray): Ground truth labels
- `out_path` (Path): Output file path

**Returns**: None (saves plot to disk)

**Calibration Method**:

- Bins predicted probabilities into 10 quantiles
- Computes fraction of positives per bin
- Plots predicted vs actual probabilities

**Perfect Calibration**: Points lie on diagonal (y = x)

**Example**:

```python
_plot_calibration_curves(results, y_test, Path('results/calibration.png'))
```

---

## Fuzzy-Monotonic Models: `src/models/fuzzy_monotonic.py`

### `train_fuzzy_monotonic_model(train_df: pd.DataFrame, test_df: pd.DataFrame, skip_monotonic: bool = False) -> Tuple[object, Dict[str, float]]`

Main entry point for fuzzy-monotonic LightGBM training.

**Parameters**:

- `train_df` (pd.DataFrame): Training set with engineered features (28 cols + `default`)
- `test_df` (pd.DataFrame): Test set with engineered features
- `skip_monotonic` (bool): If True, disables monotonic constraints (default: False)

**Returns**:

- `Tuple[model, metrics]`
  - `model` (LGBMClassifier): Trained model
  - `metrics` (Dict[str, float]): Performance metrics

**Process**:

1. Separate features and target
2. Fit fuzzy cutpoints on training set
3. Generate fuzzy membership features (21 features)
4. Build fuzzy rules (10 features)
5. Concatenate: [28 engineered] + [21 memberships] + [10 rules] = 59 features
6. Apply monotonic constraints (if enabled)
7. Train LightGBM
8. Evaluate on test set
9. Generate SHAP explanations
10. Save metrics and plots

**Hyperparameters**:

```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    monotone_constraints=constraint_vector,  # If skip_monotonic=False
    random_state=42,
    verbosity=-1
)
```

**Example**:

```python
import pandas as pd

train_df = pd.read_csv('data/processed_baseline_engineered/train.csv')
test_df = pd.read_csv('data/processed_baseline_engineered/test.csv')

# With monotonic constraints (recommended)
model, metrics = train_fuzzy_monotonic_model(train_df, test_df, skip_monotonic=False)
print(metrics)
# {
#   'roc_auc': 0.7700,
#   'pr_auc': 0.5498,
#   'brier': 0.1696,
#   'ks': 0.4144
# }

# Without constraints (for ablation study)
model_fuzzy, metrics_fuzzy = train_fuzzy_monotonic_model(
    train_df, test_df, skip_monotonic=True
)
```

**Outputs**:

- `results/metrics_fuzzy.json` (if skip_monotonic=True)
- `results/fuzzy_monotonic_metrics.json` (if skip_monotonic=False)
- `results/shap_fuzzy.png` (SHAP summary plot)

**Raises**:

- `ValueError`: If engineered features missing
- `ImportError`: If lightgbm or shap not installed

---

### `_fit_fuzzy_cutpoints(series: pd.Series) -> Dict[str, float]`

Computes percentile-based cutpoints for fuzzy membership functions.

**Parameters**:

- `series` (pd.Series): Numeric feature column from **training set**

**Returns**:

- `Dict[str, float]`: Cutpoints dictionary

**Cutpoints**:

```python
{
    'min': series.min(),
    'p25': series.quantile(0.25),
    'p33': series.quantile(0.33),
    'p50': series.quantile(0.50),
    'mid': (series.min() + series.max()) / 2,
    'p67': series.quantile(0.67),
    'p75': series.quantile(0.75),
    'max': series.max()
}
```

**Example**:

```python
import pandas as pd

train_df = pd.read_csv('data/processed_baseline_engineered/train.csv')
cutpoints = _fit_fuzzy_cutpoints(train_df['utilization'])
print(cutpoints)
# {
#   'min': 0.0,
#   'p25': 0.15,
#   'p33': 0.22,
#   'p50': 0.35,
#   'mid': 0.50,
#   'p67': 0.58,
#   'p75': 0.72,
#   'max': 3.45
# }
```

**Usage**: Cutpoints fitted on **train** set, applied to **both** train and test

---

### `_compute_memberships(values: np.ndarray, cp: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Computes Low/Medium/High fuzzy memberships.

**Parameters**:

- `values` (np.ndarray): Feature values
- `cp` (Dict[str, float]): Cutpoints from `_fit_fuzzy_cutpoints()`

**Returns**:

- `Tuple[low_mem, med_mem, high_mem]`: Three membership arrays

**Membership Functions**:

**Low (Trapezoidal)**:

```python
if x <= p25:
    return 1.0
elif p25 < x < mid:
    return (mid - x) / (mid - p25)
else:
    return 0.0
```

**Medium (Triangular)**:

```python
if p25 < x <= p50:
    return (x - p25) / (p50 - p25)
elif p50 < x < p75:
    return (p75 - x) / (p75 - p50)
else:
    return 0.0
```

**High (Trapezoidal)**:

```python
if x <= mid:
    return 0.0
elif mid < x < p75:
    return (x - mid) / (p75 - mid)
else:
    return 1.0
```

**Example**:

```python
import numpy as np

values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
cutpoints = {
    'min': 0.0, 'p25': 0.2, 'p33': 0.3, 'p50': 0.5,
    'mid': 0.5, 'p67': 0.7, 'p75': 0.8, 'max': 1.0
}

low, med, high = _compute_memberships(values, cutpoints)
print("Low:", low)   # [1.0, 0.67, 0.0, 0.0, 0.0]
print("Med:", med)   # [0.0, 0.33, 1.0, 0.33, 0.0]
print("High:", high) # [0.0, 0.0, 0.0, 0.67, 1.0]
```

**Properties**:

- All memberships ∈ [0, 1]
- Sum of memberships ≤ 1 (can be < 1 at boundaries)

---

### `_build_fuzzy_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cutpoints: Dict[str, Dict[str, float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]`

Generates fuzzy membership features for all fuzzy bases.

**Parameters**:

- `train_df` (pd.DataFrame): Training set with engineered features
- `test_df` (pd.DataFrame): Test set with engineered features
- `cutpoints` (Dict, optional): Precomputed cutpoints (if None, computed from train_df)

**Returns**:

- `Tuple[train_fuzz, test_fuzz, cutpoints_dict]`
  - `train_fuzz` (pd.DataFrame): Train with +21 fuzzy features
  - `test_fuzz` (pd.DataFrame): Test with +21 fuzzy features
  - `cutpoints_dict` (Dict): Fitted cutpoints (for saving/reuse)

**Fuzzy Bases** (7 features):

```python
fuzzy_bases = [
    'LIMIT_BAL',
    'PAY_0',
    'PAY_2',
    'utilization',
    'repay_ratio1',
    'delinquency_intensity',
    'paytrend'
]
```

**Output Features** (21 total):

```
LIMIT_BAL_low, LIMIT_BAL_med, LIMIT_BAL_high,
PAY_0_low, PAY_0_med, PAY_0_high,
PAY_2_low, PAY_2_med, PAY_2_high,
utilization_low, utilization_med, utilization_high,
repay_ratio1_low, repay_ratio1_med, repay_ratio1_high,
delinquency_intensity_low, delinquency_intensity_med, delinquency_intensity_high,
paytrend_low, paytrend_med, paytrend_high
```

**Example**:

```python
train_fuzz, test_fuzz, cutpoints = _build_fuzzy_features(train_df, test_df)
print(train_fuzz.shape)  # (24000, 49) = 28 original + 21 fuzzy
print(list(cutpoints.keys()))  # ['LIMIT_BAL', 'PAY_0', ...]

# Save cutpoints for later use
import json
with open('models/fuzzy_cutpoints.json', 'w') as f:
    json.dump(cutpoints, f)
```

**Usage**: Cutpoints fitted on train, applied to test to prevent data leakage

---

### `_build_fuzzy_rules(train_fuzz: pd.DataFrame, test_fuzz: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`

Constructs fuzzy rule features using min-AND aggregation.

**Parameters**:

- `train_fuzz` (pd.DataFrame): Training set with fuzzy memberships
- `test_fuzz` (pd.DataFrame): Test set with fuzzy memberships

**Returns**:

- `Tuple[train_rules, test_rules]`: DataFrames with +10 rule features

**Fuzzy Rules** (10 total):

```python
rules = [
    ('fuzz_rule_high_util_high_delinq',
     ['utilization_high', 'delinquency_intensity_high']),

    ('fuzz_rule_low_limit_high_delinq',
     ['LIMIT_BAL_low', 'delinquency_intensity_high']),

    ('fuzz_rule_high_pay0_low_repay',
     ['PAY_0_high', 'repay_ratio1_low']),

    ('fuzz_rule_high_util_low_repay',
     ['utilization_high', 'repay_ratio1_low']),

    ('fuzz_rule_high_delinq_worsening_trend',
     ['delinquency_intensity_high', 'paytrend_high']),

    ('fuzz_rule_low_age_high_util',
     ['AGE_low', 'utilization_high']),

    ('fuzz_rule_low_limit_low_repay',
     ['LIMIT_BAL_low', 'repay_ratio1_low']),

    ('fuzz_rule_high_pay0_high_util',
     ['PAY_0_high', 'utilization_high']),

    ('fuzz_rule_low_repay_worsening_trend',
     ['repay_ratio1_low', 'paytrend_high']),

    ('fuzz_rule_high_delinq_low_repay',
     ['delinquency_intensity_high', 'repay_ratio1_low'])
]
```

**Aggregation Logic** (min-AND):

```python
def compute_rule(row, features):
    return min([row[f] for f in features])
```

**Example**:

```python
train_rules, test_rules = _build_fuzzy_rules(train_fuzz, test_fuzz)
print(train_rules.shape)  # (24000, 59) = 49 fuzzy + 10 rules

# Inspect rule activation
sample = train_rules.iloc[0]
print(sample['fuzz_rule_high_util_high_delinq'])  # 0.75 (75% activation)
```

**Interpretation**: Rule activation = 0.85 means applicant strongly matches risk pattern

---

### Monotonic Constraints Mapping

**Constraint Dictionary**:

```python
monotone_map = {
    'LIMIT_BAL': -1,              # Higher limit → LOWER default risk
    'AGE': -1,                    # Older → LOWER risk
    'PAY_0': +1,                  # Recent late payment → HIGHER risk
    'utilization': +1,            # Higher utilization → HIGHER risk
    'repay_ratio1': -1,           # Higher repayment → LOWER risk
    'delinquency_intensity': +1,  # More delinquency → HIGHER risk
    'paytrend': -1,               # Improving trend → LOWER risk
}
```

**Constraint Vector Construction**:

```python
constraint_vector = [
    monotone_map.get(col, 0)  # 0 = unconstrained
    for col in X_train.columns
]
# Example output:
# [-1, 0, 0, 0, -1, +1, +1, -1, 0, 0, ..., +1, -1, 0, 0, ...]
```

**Usage in LightGBM**:

```python
LGBMClassifier(
    monotone_constraints=constraint_vector,
    # ... other params
)
```

---

## Ablation Study: `src/models/run_ablation.py`

### `run_all_ablations() -> pd.DataFrame`

Orchestrates full ablation study across 4 model variants.

**Parameters**: None

**Returns**:

- `pd.DataFrame`: Comparison table with metrics for all variants

**Process**:

1. **Baseline Raw**: Train baseline.py on raw features
2. **Baseline Engineered**: Train baseline.py on engineered features
3. **Fuzzy**: Train fuzzy_monotonic.py with `--skip-monotonic`
4. **Fuzzy-Monotonic**: Train fuzzy_monotonic.py (default)
5. Consolidate all metrics into `ablation_table.json`
6. Print markdown comparison table

**Outputs**:

- `results/ablation_table.json`:
  ```json
  [
    {"variant": "baseline_raw", "roc_auc": 0.7496, "pr_auc": 0.4872, ...},
    {"variant": "baseline_engineered", "roc_auc": 0.7628, "pr_auc": 0.5129, ...},
    {"variant": "fuzzy", "roc_auc": 0.7686, "pr_auc": 0.5391, ...},
    {"variant": "fuzzy_monotonic", "roc_auc": 0.7700, "pr_auc": 0.5498, ...}
  ]
  ```

**Example**:

```python
from src.models.run_ablation import run_all_ablations

results_df = run_all_ablations()
print(results_df)
#          variant  roc_auc  pr_auc  brier     ks
# 0   baseline_raw   0.7496  0.4872 0.1725 0.3821
# 1   baseline_eng   0.7628  0.5129 0.1706 0.4014
# 2          fuzzy   0.7686  0.5391 0.1696 0.4117
# 3  fuzzy_monotonic 0.7700  0.5498 0.1696 0.4144
```

**Runtime**: ~2-3 minutes

---

## Visualization: `src/models/plot_ablation_pr_auc.py`

### `plot_pr_auc_comparison(ablation_path: Path, out_path: Path) -> None`

Generates PR-AUC bar chart for ablation study.

**Parameters**:

- `ablation_path` (Path): Path to `ablation_table.json`
- `out_path` (Path): Output path for PNG (e.g., `results/ablation_pr_auc.png`)

**Returns**: None (saves plot to disk)

**Plot Features**:

- Horizontal bar chart
- Variants sorted by PR-AUC (ascending)
- Color-coded bars (gradient from low to high)
- PR-AUC values annotated on bars
- Title, axis labels, grid

**Example**:

```python
from pathlib import Path
plot_pr_auc_comparison(
    Path('results/ablation_table.json'),
    Path('results/ablation_pr_auc.png')
)
```

**Output**: PNG file with dimensions 10×6 inches, 300 DPI

---

## Interpretability: `src/models/german_interpretability.py`

### `generate_german_fuzzy_features() -> pd.DataFrame`

Generates fuzzy features for German Credit dataset.

**Parameters**: None (loads German dataset from `Data/german_credit.csv`)

**Returns**:

- `pd.DataFrame`: German credit data with fuzzy features

**Use Case**: Demonstrates interpretability on smaller, validated dataset

**Process**:

1. Load German credit dataset
2. Preprocess (encode, scale)
3. Generate fuzzy memberships
4. Build fuzzy rules
5. Save to `Data/german_fuzzy_rules.csv`

**Outputs**:

- `Data/german_fuzzy_rules.csv`
- SHAP plots for German dataset

**Example**:

```python
german_fuzz = generate_german_fuzzy_features()
print(german_fuzz.shape)  # (1000, ~40)
```

---

## Utility Functions

### `_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float`

Computes Kolmogorov-Smirnov statistic.

**Parameters**:

- `y_true` (np.ndarray): Ground truth labels
- `y_prob` (np.ndarray): Predicted probabilities

**Returns**:

- `float`: KS statistic ∈ [0, 1]

**Definition**: Maximum vertical distance between CDFs of positive and negative classes

**Example**:

```python
ks = _ks_statistic(y_test, probs)
print(f"KS: {ks:.4f}")  # 0.4144
```

**Interpretation**:

- KS > 0.4: Strong model
- KS > 0.3: Good model
- KS < 0.2: Weak model

---

## Constants & Configuration

### Fuzzy Bases

```python
FUZZY_BASES = [
    'LIMIT_BAL',
    'PAY_0',
    'PAY_2',
    'utilization',
    'repay_ratio1',
    'delinquency_intensity',
    'paytrend'
]
```

### LightGBM Hyperparameters

```python
DEFAULT_LGBM_PARAMS = {
    'is_unbalance': True,
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'random_state': 42,
    'verbosity': -1
}
```

### File Paths

```python
DATA_RAW = Path('Data/taiwan_default_of_credit_card_clients.csv')
DATA_PROCESSED_RAW = Path('data/processed_baseline_raw/')
DATA_PROCESSED_ENG = Path('data/processed_baseline_engineered/')
RESULTS_DIR = Path('results/')
```

---

## Error Handling

### Common Exceptions

**FileNotFoundError**:

```python
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Raw dataset not found: {data_path}")
```

**ValueError**:

```python
required_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', ...]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required features: {missing}")
```

**ImportError**:

```python
try:
    import lightgbm
    import shap
except ImportError as e:
    raise ImportError("Install dependencies: pip install lightgbm shap") from e
```

---

## Type Hints

### Common Types

```python
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np

# Function signatures
def func(
    df: pd.DataFrame,
    values: np.ndarray,
    config: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ...
```

---

## Testing Utilities

### Validate Monotonicity

```python
def validate_monotonicity(
    model: LGBMClassifier,
    feature_name: str,
    constraint: int,
    X: pd.DataFrame
) -> bool:
    """
    Check if model respects monotonic constraint.

    Parameters:
        model: Trained LightGBM model
        feature_name: Feature to test
        constraint: +1 or -1
        X: Sample data

    Returns:
        True if monotonicity holds, False otherwise
    """
    # Generate linearly spaced feature values
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), 100)

    # Fix other features at median
    X_test = pd.DataFrame({
        col: [X[col].median()] * 100 if col != feature_name else feature_values
        for col in X.columns
    })

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Check monotonicity
    diffs = np.diff(probs)
    if constraint == +1:
        return np.all(diffs >= -1e-6)  # Allow small numerical errors
    else:
        return np.all(diffs <= 1e-6)
```

---

## CLI Interface

### Preprocessing

```bash
python src/data/preprocess.py [--engineered]
```

### Baseline Training

```bash
python src/models/baseline.py
```

### Fuzzy-Monotonic Training

```bash
python src/models/fuzzy_monotonic.py \
  [--skip-monotonic] \
  [--variant NAME] \
  [--metrics-out PATH]
```

### Ablation Study

```bash
python src/models/run_ablation.py
```

### Visualization

```bash
python src/models/plot_ablation_pr_auc.py
```

---

## Development Guidelines

### Adding New Features

1. Update `_add_engineered_features()` in `preprocess.py`
2. Add to `fuzzy_bases` list (if applicable)
3. Add to `monotone_map` (if applicable)
4. Update tests

### Adding New Fuzzy Rules

1. Edit `_build_fuzzy_rules()` in `fuzzy_monotonic.py`
2. Append rule tuple to `rules` list
3. Validate rule activations
4. Document economic rationale

### Modifying Metrics

1. Update `_compute_metrics()` or `_metrics()`
2. Update JSON output schema
3. Update ablation table generation
4. Update visualization scripts

---

## References

For implementation details, see:

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [DATA.md](DATA.md) - Dataset documentation
- [MODELS.md](MODELS.md) - Model specifications
- [USAGE.md](USAGE.md) - Usage examples

For scientific background, see:

- `Latex/extended_ieee.tex` - Full research paper
- `Latex/elsevier_format.tex` - Journal format paper
