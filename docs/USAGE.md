# Usage Guide

## Quick Start

### Complete Pipeline (Recommended)

Run the full ablation study to reproduce all results:

```bash
# 1. Preprocess data (both raw and engineered variants)
python src/data/preprocess.py
python src/data/preprocess.py --engineered

# 2. Run complete ablation study
python src/models/run_ablation.py

# 3. Generate comparison plots
python src/models/plot_ablation_pr_auc.py
```

**Expected Output**:

```
results/
├── metrics_baseline.json
├── baseline_engineered_metrics.json
├── fuzzy_metrics.json
├── fuzzy_monotonic_metrics.json
├── ablation_table.json
├── pr_curve.png
├── calibration.png
├── shap_fuzzy.png
└── ablation_pr_auc.png
```

**Runtime**: ~2-3 minutes on standard laptop

---

## Step-by-Step Workflows

### 1. Data Preprocessing Only

**Scenario**: You want to prepare data for custom modeling

#### Raw Features (23 features)

```bash
python src/data/preprocess.py
```

**Outputs**:

- `data/processed_baseline_raw/train.csv` (24,000 samples)
- `data/processed_baseline_raw/test.csv` (6,000 samples)

**Contents**: Original 23 features, scaled, encoded, stratified split

#### Engineered Features (28 features)

```bash
python src/data/preprocess.py --engineered
```

**Outputs**:

- `data/processed_baseline_engineered/train.csv` (24,000 samples)
- `data/processed_baseline_engineered/test.csv` (6,000 samples)

**Contents**: 23 original + 5 engineered features

**Engineered Features Added**:

- `BILL_AMT_AVG`: Average bill amount
- `utilization`: Credit utilization ratio
- `repay_ratio1`: Recent repayment ratio
- `delinquency_intensity`: Cumulative payment delay severity
- `paytrend`: Payment behavior trend

---

### 2. Train Baseline Models

**Scenario**: Establish baseline performance without fuzzy logic

#### Prerequisites

```bash
# Ensure preprocessed data exists
python src/data/preprocess.py  # For raw baseline
```

#### Execution

```bash
python src/models/baseline.py
```

#### Process

1. Loads `data/processed_baseline_raw/train.csv` and `test.csv`
2. Trains Logistic Regression (balanced)
3. Trains LightGBM (balanced)
4. Compares both models
5. Selects best model (typically LightGBM)

#### Outputs

- `results/metrics_baseline.json`:
  ```json
  {
    "roc_auc": 0.7496,
    "pr_auc": 0.4872,
    "brier": 0.1725,
    "ks": 0.3821
  }
  ```
- `results/pr_curve.png`: Precision-Recall curves for both models
- `results/calibration.png`: Calibration curves for both models

#### Interpreting Results

- **ROC-AUC > 0.75**: Good discriminative ability
- **PR-AUC ~0.49**: Room for improvement on minority class
- **Brier < 0.20**: Acceptable calibration
- **KS > 0.38**: Strong separation between classes

---

### 3. Train Fuzzy Model (No Monotonic Constraints)

**Scenario**: Evaluate fuzzy logic impact without constraints

#### Prerequisites

```bash
# Ensure engineered data exists
python src/data/preprocess.py --engineered
```

#### Execution

```bash
python src/models/fuzzy_monotonic.py \
  --skip-monotonic \
  --variant fuzzy \
  --metrics-out results/fuzzy_metrics.json
```

#### Parameters

- `--skip-monotonic`: Disables monotonic constraints
- `--variant fuzzy`: Names the variant (for logging)
- `--metrics-out`: Custom metrics output path

#### Process

1. Loads engineered train/test data
2. Fits fuzzy cutpoints on training set
3. Generates 21 fuzzy membership features
4. Builds 10 fuzzy rule features
5. Trains LightGBM on 59 total features
6. Computes SHAP explanations
7. Saves metrics and plots

#### Outputs

- `results/fuzzy_metrics.json`: Performance metrics
- `results/shap_fuzzy.png`: SHAP summary plot
- Console: Feature importance ranking

#### Expected Performance

```
ROC-AUC: 0.7686 (+2.5% vs Baseline)
PR-AUC:  0.5391 (+10.7% vs Baseline)
Brier:   0.1696 (−1.7% vs Baseline)
KS:      0.4117 (+7.7% vs Baseline)
```

---

### 4. Train Fuzzy-Monotonic Model (Recommended)

**Scenario**: Full framework with regulatory compliance

#### Prerequisites

```bash
# Ensure engineered data exists
python src/data/preprocess.py --engineered
```

#### Execution

```bash
python src/models/fuzzy_monotonic.py \
  --variant fuzzy_monotonic \
  --metrics-out results/fuzzy_monotonic_metrics.json
```

#### Parameters

- `--variant fuzzy_monotonic`: Model variant name
- `--metrics-out`: Output path for metrics
- **No** `--skip-monotonic`: Monotonic constraints enabled by default

#### Process

Same as Fuzzy Model, but with 7 monotonic constraints applied:

- `LIMIT_BAL` → −1 (higher limit = lower risk)
- `AGE` → −1 (older = lower risk)
- `PAY_0` → +1 (recent late = higher risk)
- `utilization` → +1 (higher util = higher risk)
- `repay_ratio1` → −1 (higher repay = lower risk)
- `delinquency_intensity` → +1 (more delinq = higher risk)
- `paytrend` → −1 (improving = lower risk)

#### Outputs

- `results/fuzzy_monotonic_metrics.json`: **Best performance metrics**
- `results/shap_fuzzy_monotonic.png`: SHAP explanations (if generated)
- Console: Monotonic constraint validation report

#### Expected Performance

```
ROC-AUC: 0.7700 (+2.7% vs Baseline)
PR-AUC:  0.5498 (+12.9% vs Baseline) ← BEST
Brier:   0.1696 (−1.7% vs Baseline)
KS:      0.4144 (+8.5% vs Baseline)
```

---

### 5. Run Full Ablation Study

**Scenario**: Compare all model variants systematically

#### Prerequisites

Ensure both raw and engineered preprocessed data exist:

```bash
python src/data/preprocess.py
python src/data/preprocess.py --engineered
```

#### Execution

```bash
python src/models/run_ablation.py
```

#### Process

Orchestrates 4 model training runs:

1. **Baseline Raw**: Logistic Regression + LightGBM on 23 features
2. **Baseline Engineered**: LightGBM on 28 features
3. **Fuzzy**: LightGBM on 59 features (no constraints)
4. **Fuzzy-Monotonic**: LightGBM on 59 features (with constraints)

#### Outputs

- Individual metrics files (as above)
- `results/ablation_table.json`:
  ```json
  [
    {"variant": "baseline_raw", "roc_auc": 0.7496, "pr_auc": 0.4872, ...},
    {"variant": "baseline_engineered", "roc_auc": 0.7628, "pr_auc": 0.5129, ...},
    {"variant": "fuzzy", "roc_auc": 0.7686, "pr_auc": 0.5391, ...},
    {"variant": "fuzzy_monotonic", "roc_auc": 0.7700, "pr_auc": 0.5498, ...}
  ]
  ```
- Console: Markdown comparison table

#### Console Output Example

```
| Variant              | ROC-AUC | PR-AUC | Brier  | KS     |
|----------------------|---------|--------|--------|--------|
| baseline_raw         | 0.7496  | 0.4872 | 0.1725 | 0.3821 |
| baseline_engineered  | 0.7628  | 0.5129 | 0.1706 | 0.4014 |
| fuzzy                | 0.7686  | 0.5391 | 0.1696 | 0.4117 |
| fuzzy_monotonic      | 0.7700  | 0.5498 | 0.1696 | 0.4144 |
```

---

### 6. Generate Visualizations

**Scenario**: Create publication-ready plots

#### PR-AUC Comparison Bar Chart

```bash
python src/models/plot_ablation_pr_auc.py
```

**Output**: `results/ablation_pr_auc.png`

**Description**: Horizontal bar chart comparing PR-AUC across all variants

#### Baseline Model Comparisons

Already generated during baseline training:

- `results/pr_curve.png`: Logistic Regression vs LightGBM PR curves
- `results/calibration.png`: Calibration reliability diagrams

#### SHAP Explanations

Generated during fuzzy/fuzzy-monotonic training:

- `results/shap_fuzzy.png`: Global feature importance (beeswarm plot)
- Individual waterfall plots (if implemented)

---

## Advanced Usage

### Custom Fuzzy Bases

**Scenario**: Add new features to fuzzy transformation

#### Edit `fuzzy_monotonic.py`

```python
# Line ~120
fuzzy_bases = [
    'LIMIT_BAL',
    'PAY_0',
    'PAY_2',
    'utilization',
    'repay_ratio1',
    'delinquency_intensity',
    'paytrend',
    'YOUR_NEW_FEATURE'  # Add here
]
```

#### Add Monotonic Constraint (if applicable)

```python
# Line ~180
monotone_map = {
    'LIMIT_BAL': -1,
    # ... existing constraints
    'YOUR_NEW_FEATURE': +1  # or -1, or omit for unconstrained
}
```

#### Retrain

```bash
python src/models/fuzzy_monotonic.py
```

---

### Custom Fuzzy Rules

**Scenario**: Add domain-expert risk patterns

#### Edit `_build_fuzzy_rules()` in `fuzzy_monotonic.py`

```python
# Line ~95
rules = [
    # Existing rules...
    ('fuzz_rule_high_delinq_low_repay',
     ['delinquency_intensity_high', 'repay_ratio1_low']),

    # Add your new rule
    ('fuzz_rule_your_pattern',
     ['FEATURE1_high', 'FEATURE2_low', 'FEATURE3_med'])
]
```

**Rule Logic**: Uses **min-AND** aggregation

```python
activation = min([
    row['FEATURE1_high'],
    row['FEATURE2_low'],
    row['FEATURE3_med']
])
```

#### Retrain

```bash
python src/models/fuzzy_monotonic.py
```

---

### Hyperparameter Tuning

**Scenario**: Optimize LightGBM hyperparameters

#### Manual Grid Search

Create `tune_hyperparameters.py`:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier
import pandas as pd

# Load data
X_train = pd.read_csv('data/processed_baseline_engineered/train.csv')
y_train = X_train.pop('default')

# Define grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'num_leaves': [15, 31, 63]
}

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LGBMClassifier(is_unbalance=True, random_state=42)

# Search
search = GridSearchCV(
    clf, param_grid,
    scoring='average_precision',  # PR-AUC
    cv=cv,
    n_jobs=-1
)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best PR-AUC:", search.best_score_)
```

#### Execute

```bash
python tune_hyperparameters.py
```

---

### Custom Metrics

**Scenario**: Add business-specific metrics (e.g., expected loss)

#### Edit `_metrics()` function

```python
# In baseline.py or fuzzy_monotonic.py

def _metrics(y_true, y_prob, threshold=0.5):
    """Compute all metrics"""
    from sklearn.metrics import roc_auc_score, brier_score_loss, ...

    # Existing metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ks = _ks_statistic(y_true, y_prob)

    # Custom metric: Expected Loss
    y_pred = (y_prob > threshold).astype(int)
    loss_per_default = 10000  # Example: $10k loss per default
    expected_loss = (y_pred * y_true).sum() * loss_per_default

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'ks': ks,
        'expected_loss': expected_loss  # New metric
    }
```

---

### Inference on New Data

**Scenario**: Predict default probability for new applicant

#### Prepare New Data

```python
import pandas as pd

new_applicant = pd.DataFrame([{
    'LIMIT_BAL': 50000,
    'SEX': 1,
    'EDUCATION': 2,
    'MARRIAGE': 1,
    'AGE': 30,
    'PAY_0': 0,
    'PAY_2': 0,
    # ... all 23 original features
}])
```

#### Option 1: Full Pipeline (Recommended)

```python
import joblib
from src.data.preprocess import _add_engineered_features
from src.models.fuzzy_monotonic import _build_fuzzy_features, _build_fuzzy_rules

# 1. Engineer features
new_applicant = _add_engineered_features(new_applicant)

# 2. Load artifacts
model = joblib.load('models/fuzzy_monotonic_v1.pkl')
scaler = joblib.load('models/preprocessor_v1.pkl')
cutpoints = joblib.load('models/fuzzy_cutpoints_v1.json')

# 3. Generate fuzzy features
new_fuzz, _, _ = _build_fuzzy_features(
    new_applicant, new_applicant, cutpoints
)
new_rules = _build_fuzzy_rules(new_fuzz, new_fuzz)

# 4. Predict
prob = model.predict_proba(new_rules)[0, 1]
print(f"Default Probability: {prob:.2%}")
```

#### Option 2: Quick Script

Save as `inference.py`:

```python
import sys
import pandas as pd
from src.models.fuzzy_monotonic import train_fuzzy_monotonic_model

# Load trained model artifacts (requires saving in training script)
# model, scaler, cutpoints = load_artifacts()

# Or retrain (not recommended for production)
# For demo only:
train_df = pd.read_csv('data/processed_baseline_engineered/train.csv')
test_df = pd.read_csv('data/processed_baseline_engineered/test.csv')
model, metrics = train_fuzzy_monotonic_model(
    train_df, test_df, skip_monotonic=False
)

# New data (passed as JSON string)
import json
new_data = json.loads(sys.argv[1])
new_df = pd.DataFrame([new_data])

# Preprocess + predict (implement full pipeline)
# prob = predict_pipeline(model, new_df, scaler, cutpoints)
# print(prob)
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'lightgbm'

**Cause**: Missing dependency

**Solution**:

```bash
pip install lightgbm
# Or install all dependencies
pip install -r requirements.txt
```

---

#### 2. FileNotFoundError: data/processed_baseline_raw/train.csv

**Cause**: Preprocessing not run

**Solution**:

```bash
python src/data/preprocess.py
```

---

#### 3. ValueError: Engineered feature missing: utilization

**Cause**: Trying to run fuzzy model without engineered features

**Solution**:

```bash
# Regenerate engineered data
python src/data/preprocess.py --engineered
```

---

#### 4. SHAP plots not generated

**Cause**: Large dataset slows SHAP computation

**Solution**:

- Reduce sample size in `fuzzy_monotonic.py`:
  ```python
  # Line ~250
  sample_size = 500  # Reduce to 100 for faster execution
  ```
- Or skip SHAP:
  ```python
  # Comment out SHAP code block
  ```

---

#### 5. Monotonic constraint violation warnings

**Cause**: LightGBM may violate constraints with small datasets

**Solution**:

- Increase `min_child_samples`:
  ```python
  LGBMClassifier(
      min_child_samples=30,  # Default 20
      ...
  )
  ```
- Validate manually using partial dependence plots

---

#### 6. Memory Error on large datasets

**Cause**: Fuzzy features triple feature count

**Solution**:

- Use feature selection to reduce fuzzy bases
- Subsample training data:
  ```python
  train_df = train_df.sample(frac=0.5, random_state=42)
  ```

---

## Best Practices

### Development Workflow

1. **Start Simple**: Run baseline first to validate pipeline
2. **Incremental Changes**: Add one component at a time (engineered → fuzzy → monotonic)
3. **Version Results**: Save metrics to dated directories
   ```bash
   mkdir -p results/2024-01-15/
   python src/models/run_ablation.py
   cp results/*.json results/2024-01-15/
   ```
4. **Document Changes**: Keep log of modifications to fuzzy rules, hyperparameters

### Production Checklist

- [ ] Persist trained models (`joblib.dump`)
- [ ] Save preprocessing artifacts (scaler, encoder, cutpoints)
- [ ] Version control model artifacts (Git LFS or DVC)
- [ ] Implement API wrapper (FastAPI/Flask)
- [ ] Add input validation (Pydantic models)
- [ ] Monitor data drift (Evidently AI)
- [ ] Set up retraining pipeline (monthly/quarterly)
- [ ] Configure logging (Python logging module)
- [ ] Write unit tests (pytest)
- [ ] Document model card (model version, metrics, limitations)

### Collaboration

**Sharing Results**:

```bash
# Create shareable archive
tar -czvf results_archive.tar.gz results/ data/processed_*/

# Or use Git
git add results/*.json results/*.png
git commit -m "Update ablation study results"
git push
```

**Reproducibility**:

```bash
# Document environment
pip freeze > requirements.txt

# Document exact commands
echo "python src/data/preprocess.py --engineered" > run.sh
echo "python src/models/run_ablation.py" >> run.sh
chmod +x run.sh
```

---

## Command Reference

### Complete Command List

```bash
# Data Preprocessing
python src/data/preprocess.py                    # Raw features
python src/data/preprocess.py --engineered       # + Engineered features

# Baseline Models
python src/models/baseline.py                    # Train baseline models

# Fuzzy Models
python src/models/fuzzy_monotonic.py \
  --skip-monotonic \
  --variant fuzzy \
  --metrics-out results/fuzzy_metrics.json       # Fuzzy (no constraints)

python src/models/fuzzy_monotonic.py \
  --variant fuzzy_monotonic \
  --metrics-out results/fuzzy_monotonic_metrics.json  # Fuzzy-Monotonic

# Ablation Study
python src/models/run_ablation.py                # Compare all variants

# Visualization
python src/models/plot_ablation_pr_auc.py        # PR-AUC bar chart

# German Interpretability (optional)
python src/models/german_interpretability.py     # SHAP on German dataset
```

### Argument Reference

#### `preprocess.py`

- `--engineered`: Generate 5 engineered features

#### `fuzzy_monotonic.py`

- `--skip-monotonic`: Disable monotonic constraints
- `--variant NAME`: Set variant name (default: "fuzzy_monotonic")
- `--metrics-out PATH`: Custom metrics output path (default: "results/metrics_fuzzy.json")

---

## Jupyter Notebook Workflows

### Interactive Exploration

Create `notebooks/explore.ipynb`:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
train = pd.read_csv('../data/processed_baseline_engineered/train.csv')

# Exploratory analysis
train['default'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.show()

# Feature correlations
train.corr()['default'].sort_values(ascending=False)
```

### Model Training Notebook

Create `notebooks/train_model.ipynb`:

```python
from src.models.fuzzy_monotonic import train_fuzzy_monotonic_model
import pandas as pd

# Load data
train_df = pd.read_csv('../data/processed_baseline_engineered/train.csv')
test_df = pd.read_csv('../data/processed_baseline_engineered/test.csv')

# Train
model, metrics = train_fuzzy_monotonic_model(
    train_df, test_df, skip_monotonic=False
)

# Results
print(metrics)
```

---

## FAQ

**Q: Which model should I use in production?**  
A: **Fuzzy-Monotonic** (best performance + regulatory compliance)

**Q: How do I update fuzzy cutpoints for new data?**  
A: Refit on new training data using `_fit_fuzzy_cutpoints()`, save to JSON

**Q: Can I use this framework on other datasets?**  
A: Yes, but requires modifying feature engineering and fuzzy rules to match domain

**Q: How to explain a specific prediction?**  
A: Use SHAP waterfall plots (see MODELS.md § Explainability)

**Q: What if monotonic constraints hurt performance?**  
A: Validate constraints make economic sense; if not, use Fuzzy model without constraints
