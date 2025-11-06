# Architecture Guide

## System Overview

The Fuzzy-Monotonic LightGBM framework implements a modular, pipeline-based architecture that separates data processing, feature engineering, model training, and evaluation into distinct components.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ Taiwan CSV   │         │ German CSV   │                      │
│  │ (30,000)     │         │ (1,000)      │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
└─────────┼────────────────────────┼──────────────────────────────┘
          │                        │
          ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessing Layer                           │
│  ┌───────────────────────────────────────────────────┐          │
│  │  src/data/preprocess.py                           │          │
│  │  • Read & clean raw CSV                           │          │
│  │  • Feature engineering (engineered mode)          │          │
│  │  • Encoding (Label, One-Hot)                      │          │
│  │  • Scaling (RobustScaler)                         │          │
│  │  • Train/test split (80/20 stratified)            │          │
│  └───────────────────┬───────────────────────────────┘          │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Processed Data Store                           │
│  ┌────────────────────────┐  ┌────────────────────────┐         │
│  │ processed_baseline_raw/│  │ processed_baseline_    │         │
│  │  - train.csv           │  │   engineered/          │         │
│  │  - test.csv            │  │  - train.csv           │         │
│  └────────────────────────┘  │  - test.csv            │         │
│                               └────────────────────────┘         │
└──────────────────────┬───────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐          ┌─────────────────────────────────┐
│  Baseline Layer  │          │  Fuzzy-Monotonic Layer          │
│                  │          │                                 │
│ baseline.py      │          │ fuzzy_monotonic.py              │
│                  │          │                                 │
│ ┌──────────────┐ │          │ ┌─────────────────────────────┐ │
│ │ Logistic     │ │          │ │ Fuzzy Membership Generator  │ │
│ │ Regression   │ │          │ │ • Percentile-based cutpoints│ │
│ │ (balanced)   │ │          │ │ • Low/Med/High memberships  │ │
│ └──────────────┘ │          │ └──────────────┬──────────────┘ │
│                  │          │                │                 │
│ ┌──────────────┐ │          │ ┌──────────────▼──────────────┐ │
│ │ LightGBM     │ │          │ │ Fuzzy Rule Builder          │ │
│ │ (balanced)   │ │          │ │ • min-AND activations       │ │
│ └──────────────┘ │          │ │ • 10 predefined rules       │ │
└──────┬───────────┘          │ └──────────────┬──────────────┘ │
       │                      │                │                 │
       │                      │ ┌──────────────▼──────────────┐ │
       │                      │ │ Monotonic LightGBM          │ │
       │                      │ │ • Economic constraints      │ │
       │                      │ │ • Class balanced            │ │
       │                      │ └──────────────┬──────────────┘ │
       │                      └────────────────┼─────────────────┘
       │                                       │
       ▼                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Evaluation & Explainability Layer              │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ Metrics        │  │ Visualization  │  │ Explainability    │  │
│  │ • ROC-AUC      │  │ • PR curves    │  │ • SHAP (Tree)     │  │
│  │ • PR-AUC       │  │ • Calibration  │  │ • Fuzzy rules     │  │
│  │ • Brier        │  │ • SHAP plots   │  │ • Monotonic check │  │
│  │ • KS           │  └────────────────┘  └───────────────────┘  │
│  └────────────────┘                                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  Results Output    │
                  │  • metrics JSON    │
                  │  • PNG plots       │
                  │  • Comparison table│
                  └────────────────────┘
```

## Component Details

### 1. Data Layer

**Location**: `Data/`, `src/data/`

**Responsibilities**:

- Raw dataset storage
- Dataset loading utilities
- Basic data validation

**Files**:

- `Data/taiwan_default_of_credit_card_clients.csv`: Primary dataset
- `Data/german_credit.csv`: Interpretability demonstration dataset

### 2. Preprocessing Layer

**Location**: `src/data/preprocess.py`

**Responsibilities**:

- Feature engineering
- Data cleaning
- Encoding categorical variables
- Scaling numeric features
- Train/test splitting

**Key Functions**:

```python
def _read_taiwan_csv(data_path: Path) -> pd.DataFrame
def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame
def _apply_scaling(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
def main() -> None
```

**CLI Interface**:

```bash
python src/data/preprocess.py [--engineered]
```

**Outputs**:

- `data/processed_baseline_raw/`: Original features only
- `data/processed_baseline_engineered/`: + engineered features

### 3. Baseline Model Layer

**Location**: `src/models/baseline.py`

**Responsibilities**:

- Train baseline models (Logistic Regression, LightGBM)
- Evaluate performance
- Generate comparison visualizations

**Key Functions**:

```python
def train_baseline_models() -> Tuple[object, Dict[str, float]]
def _compute_metrics(y_true, y_prob) -> Dict[str, float]
def _plot_pr_curves(results, y_true, out_path) -> None
def _plot_calibration_curves(results, y_true, out_path) -> None
```

**Outputs**:

- `results/metrics_baseline.json`
- `results/pr_curve.png`
- `results/calibration.png`

### 4. Fuzzy-Monotonic Layer

**Location**: `src/models/fuzzy_monotonic.py`

**Responsibilities**:

- Generate fuzzy memberships
- Build fuzzy rules
- Train monotonic LightGBM
- Generate SHAP explanations

**Key Components**:

#### 4.1 Fuzzy Membership Generator

```python
def _fit_fuzzy_cutpoints(series: pd.Series) -> Dict[str, float]
def _compute_memberships(values, cp) -> Tuple[ndarray, ndarray, ndarray]
def _build_fuzzy_features(...) -> Tuple[DataFrame, DataFrame, Dict]
```

Generates Low/Med/High memberships using:

- **Low**: Trapezoid declining from (min, p25, p33, mid)
- **Medium**: Triangle at (p25, p50, p75)
- **High**: Trapezoid rising from (mid, p67, p75, max)

#### 4.2 Fuzzy Rule Builder

```python
def _build_fuzzy_rules(train_m, test_m) -> Tuple[DataFrame, DataFrame]
```

Implements 10 domain-expert rules:

```python
rules = [
    ('fuzz_rule_high_util_high_delinq', ['utilization_high', 'delinquency_intensity_high']),
    ('fuzz_rule_low_limit_high_delinq', ['LIMIT_BAL_low', 'delinquency_intensity_high']),
    ('fuzz_rule_high_pay0_low_repay', ['PAY_0_high', 'repay_ratio1_low']),
    ...
]
```

#### 4.3 Monotonic Constraint Engine

```python
monotone_map = {
    'LIMIT_BAL': -1,          # Higher limit → lower risk
    'AGE': -1,                # Older → lower risk
    'PAY_0': +1,              # Recent late → higher risk
    'utilization': +1,        # High util → higher risk
    'repay_ratio1': -1,       # Higher repay → lower risk
    'delinquency_intensity': +1,  # More delinq → higher risk
    'paytrend': -1,           # Improving → lower risk
}
```

**CLI Interface**:

```bash
python src/models/fuzzy_monotonic.py [--skip-monotonic] [--variant NAME] [--metrics-out PATH]
```

**Outputs**:

- `results/metrics_fuzzy.json`
- `results/shap_fuzzy.png`

### 5. Ablation Study Orchestrator

**Location**: `src/models/run_ablation.py`

**Responsibilities**:

- Execute all model variants
- Collect metrics
- Generate comparison tables

**Workflow**:

```python
1. Preprocess (raw) → baseline_raw_metrics
2. Preprocess (engineered) → baseline_engineered_metrics
3. Fuzzy (no monotonic) → fuzzy_metrics
4. Fuzzy + Monotonic → fuzzy_monotonic_metrics
5. Consolidate → ablation_table.json
6. Print markdown table
```

**Outputs**:

- `results/ablation_table.json`
- Console: Formatted markdown table

### 6. Visualization Layer

**Location**: `src/models/plot_ablation_pr_auc.py`

**Responsibilities**:

- Generate PR-AUC comparison bar chart

**Outputs**:

- `results/ablation_pr_auc.png`

## Data Flow

### Training Pipeline

```
1. Raw CSV → Preprocess → Processed Train/Test
                ↓
2. Processed → Baseline Models → Metrics + Plots
                ↓
3. Raw + Processed → Fuzzy Features → Combined Features
                ↓
4. Combined → Fuzzy-Monotonic Model → Metrics + SHAP
                ↓
5. All Metrics → Ablation Comparison → Table + Plots
```

### Inference Pipeline (Future)

```
1. New Applicant Data → Preprocess (same scaler)
                ↓
2. Engineered Features
                ↓
3. Fuzzy Memberships (same cutpoints)
                ↓
4. Fuzzy Rules (same activation logic)
                ↓
5. Model Prediction → Default Probability + Explanation
```

## Design Patterns

### 1. Pipeline Pattern

Each module outputs to disk, enabling:

- Reproducibility
- Debugging
- Intermediate inspection

### 2. Separation of Concerns

- **Data**: preprocess.py
- **Baseline**: baseline.py
- **Advanced**: fuzzy_monotonic.py
- **Orchestration**: run_ablation.py
- **Visualization**: plot_ablation_pr_auc.py

### 3. Configuration via CLI

All scripts accept command-line arguments for flexibility:

```bash
--engineered     # Feature engineering flag
--skip-monotonic # Disable monotonic constraints
--variant        # Model variant name
--metrics-out    # Custom metrics output path
```

### 4. JSON for Metrics

Standardized metrics format:

```json
{
  "roc_auc": 0.77,
  "pr_auc": 0.5498,
  "brier": 0.1696,
  "ks": 0.4144
}
```

## Extension Points

### Adding New Features

1. Modify `_add_engineered_features()` in `preprocess.py`
2. Update fuzzy bases list in `fuzzy_monotonic.py`
3. Add monotonic constraint to `monotone_map`

### Adding New Fuzzy Rules

1. Edit `_build_fuzzy_rules()` in `fuzzy_monotonic.py`
2. Add tuple to `rules` list with feature combinations

### Adding New Models

1. Create new module in `src/models/`
2. Implement `train_*_model()` function
3. Add to `run_ablation.py` workflow

### Custom Metrics

1. Extend `_metrics()` or `_compute_metrics()`
2. Update JSON output format
3. Modify ablation table generation

## Performance Considerations

### Memory

- Processed data cached on disk
- SHAP uses sample (n=500) for speed
- Fuzzy features appended, not replaced

### Computation

- LightGBM leverages multi-threading (`n_jobs=-1`)
- RobustScaler fit once on train
- Fuzzy cutpoints computed once

### Scalability

- Current: 30K samples
- Tested: Up to 100K samples
- Bottleneck: SHAP computation (O(n·trees))

## Dependencies

### Required

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `scikit-learn >= 1.0.0`
- `lightgbm >= 3.3.0`
- `matplotlib >= 3.5.0`
- `shap >= 0.41.0`

### Optional

- `seaborn >= 0.11.0` (enhanced plots)
- `jupyter >= 1.0.0` (notebooks)

## Error Handling

### Missing Features

```python
engineered_cols = ["BILL_AMT_AVG", "utilization", ...]
for col in engineered_cols:
    if col not in X_train.columns:
        raise ValueError(f"Engineered feature missing: {col}")
```

### Import Guards

```python
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("lightgbm is required...") from e
```

### Path Resolution

```python
def _resolve_paths():
    repo_root = Path(__file__).resolve().parents[2]
    # Ensures consistent paths regardless of CWD
```

## Testing Strategy

### Unit Tests (Recommended)

- Test feature engineering functions
- Validate fuzzy membership calculations
- Check monotonic constraint mapping

### Integration Tests

- End-to-end pipeline execution
- Metrics validation (ranges)
- Output file existence

### Validation

- Cross-validation on Taiwan dataset
- Hold-out set evaluation
- Temporal validation (if time-series split)

## Deployment Considerations

### Production Checklist

- [ ] Persist trained model (pickle/joblib)
- [ ] Save preprocessing artifacts (scaler, encoder)
- [ ] Save fuzzy cutpoints JSON
- [ ] Version control model artifacts
- [ ] API wrapper for inference
- [ ] Monitoring dashboard
- [ ] Drift detection
- [ ] Model retraining pipeline

### Model Artifacts

```
models/
├── fuzzy_monotonic_v1.pkl        # Trained model
├── preprocessor_v1.pkl            # RobustScaler + LabelEncoder
├── fuzzy_cutpoints_v1.json        # Percentile cutpoints
└── metadata_v1.json               # Version, date, metrics
```

### API Example (Future)

```python
from api import CreditRiskAPI

api = CreditRiskAPI(model_version="v1")
result = api.predict(applicant_data)
# {
#   "default_probability": 0.234,
#   "risk_category": "Medium",
#   "explanations": [...],
#   "fuzzy_activations": {...}
# }
```
