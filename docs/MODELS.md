# Models Documentation

## Model Variants Overview

This framework implements four model variants in an ablation study to evaluate the impact of feature engineering, fuzzy logic, and monotonic constraints:

| Variant             | Features    | Fuzzy Logic | Monotonic | Algorithm                      | PR-AUC     |
| ------------------- | ----------- | ----------- | --------- | ------------------------------ | ---------- |
| Baseline Raw        | 23 original | ❌          | ❌        | Logistic Regression + LightGBM | 0.4872     |
| Baseline Engineered | 28 (23+5)   | ❌          | ❌        | LightGBM                       | 0.5129     |
| Fuzzy               | 59 (28+31)  | ✅          | ❌        | LightGBM                       | 0.5391     |
| **Fuzzy-Monotonic** | 59 (28+31)  | ✅          | ✅        | LightGBM                       | **0.5498** |

## 1. Baseline Raw

### Description

Trains two baseline models using **only original 23 features** without feature engineering:

- Logistic Regression (balanced)
- LightGBM (balanced)

### Location

`src/models/baseline.py`

### Architecture

**Logistic Regression**:

```python
LogisticRegression(
    class_weight='balanced',    # Handles class imbalance
    max_iter=1000,              # Convergence iterations
    solver='lbfgs',             # Optimization algorithm
    random_state=42
)
```

**LightGBM**:

```python
LGBMClassifier(
    is_unbalance=True,          # Auto class weighting
    n_estimators=100,           # Number of trees
    learning_rate=0.05,         # Step size
    max_depth=7,                # Tree depth
    num_leaves=31,              # Complexity control
    random_state=42,
    verbosity=-1
)
```

### Training

**Command**:

```bash
python src/models/baseline.py
```

**Data**: `data/processed_baseline_raw/train.csv`

**Process**:

1. Load preprocessed train/test splits
2. Train Logistic Regression → compute metrics
3. Train LightGBM → compute metrics
4. Compare models → plot PR curves & calibration
5. Save best model metrics

### Outputs

**Files**:

- `results/metrics_baseline.json` - Performance metrics
- `results/pr_curve.png` - Precision-Recall curves comparison
- `results/calibration.png` - Calibration curves comparison

**Metrics Structure**:

```json
{
  "roc_auc": 0.7496,
  "pr_auc": 0.4872,
  "brier": 0.1725,
  "ks": 0.3821
}
```

### Performance

**Test Set Results**:

- **ROC-AUC**: 0.7496
- **PR-AUC**: 0.4872
- **Brier Score**: 0.1725
- **KS Statistic**: 0.3821

**Interpretation**:

- Solid baseline but limited by feature space
- Logistic regression provides interpretability
- LightGBM captures non-linearities better

## 2. Baseline Engineered

### Description

LightGBM trained on **28 features** (23 original + 5 engineered), no fuzzy logic or constraints.

### Feature Engineering

**Additional Features** (see DATA.md for details):

1. `BILL_AMT_AVG` - Average bill amount
2. `utilization` - Credit utilization ratio
3. `repay_ratio1` - Recent repayment ratio
4. `delinquency_intensity` - Cumulative payment delay severity
5. `paytrend` - Payment behavior trend

### Training

**Command**:

```bash
python src/data/preprocess.py --engineered  # Generate features
python src/models/baseline.py               # Train model
```

**Data**: `data/processed_baseline_engineered/train.csv`

### Hyperparameters

Same as Baseline Raw LightGBM:

```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42
)
```

### Performance

**Test Set Results**:

- **ROC-AUC**: 0.7628 (+1.3%)
- **PR-AUC**: 0.5129 (+5.3%)
- **Brier Score**: 0.1706 (−1.1%)
- **KS Statistic**: 0.4014 (+5.1%)

**Interpretation**:

- Engineered features improve all metrics
- **Largest gain in PR-AUC** (critical for imbalanced data)
- Demonstrates value of domain knowledge

## 3. Fuzzy Model

### Description

LightGBM trained on **59 features** (28 engineered + 21 fuzzy memberships + 10 fuzzy rules), **without monotonic constraints**.

### Location

`src/models/fuzzy_monotonic.py` (with `--skip-monotonic` flag)

### Fuzzy Feature Generation

#### Step 1: Fit Cutpoints (Training Set)

For each fuzzy base feature, compute percentiles:

```python
cutpoints = {
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

**Fuzzy Bases** (7 features):

- `LIMIT_BAL`
- `PAY_0`
- `PAY_2`
- `utilization`
- `repay_ratio1`
- `delinquency_intensity`
- `paytrend`

#### Step 2: Compute Memberships

**Low Membership** (Trapezoidal):

```python
def _low_trapezoid(x, cp):
    if x <= cp['p25']:
        return 1.0
    elif cp['p25'] < x < cp['mid']:
        return (cp['mid'] - x) / (cp['mid'] - cp['p25'])
    else:
        return 0.0
```

**Medium Membership** (Triangular):

```python
def _med_triangle(x, cp):
    if cp['p25'] < x <= cp['p50']:
        return (x - cp['p25']) / (cp['p50'] - cp['p25'])
    elif cp['p50'] < x < cp['p75']:
        return (cp['p75'] - x) / (cp['p75'] - cp['p50'])
    else:
        return 0.0
```

**High Membership** (Trapezoidal):

```python
def _high_trapezoid(x, cp):
    if x <= cp['mid']:
        return 0.0
    elif cp['mid'] < x < cp['p75']:
        return (x - cp['mid']) / (cp['p75'] - cp['mid'])
    else:
        return 1.0
```

**Output**: 7 bases × 3 memberships = **21 fuzzy membership features**

#### Step 3: Build Fuzzy Rules

Using **min-AND** aggregation:

```python
def _compute_rule_activation(row, feature_names):
    """Computes min of specified fuzzy memberships"""
    return min([row[fname] for fname in feature_names])
```

**Example Rule**:

```python
# Rule: High utilization AND high delinquency intensity
fuzz_rule_high_util_high_delinq = min(
    row['utilization_high'],
    row['delinquency_intensity_high']
)
```

**All Rules** (10 total):

1. `fuzz_rule_high_util_high_delinq` - High risk: maxed out credit + late payments
2. `fuzz_rule_low_limit_high_delinq` - Low limit + high delinquency
3. `fuzz_rule_high_pay0_low_repay` - Recent late payment + low repayment
4. `fuzz_rule_high_util_low_repay` - High utilization + poor repayment
5. `fuzz_rule_high_delinq_worsening_trend` - Increasing delinquency over time
6. `fuzz_rule_low_age_high_util` - Young + high utilization
7. `fuzz_rule_low_limit_low_repay` - Low limit + low repayment
8. `fuzz_rule_high_pay0_high_util` - Recent late + high utilization
9. `fuzz_rule_low_repay_worsening_trend` - Poor repayment + worsening
10. `fuzz_rule_high_delinq_low_repay` - High delinquency + low repayment

**Output**: **10 fuzzy rule features**

### Training

**Command**:

```bash
python src/models/fuzzy_monotonic.py --skip-monotonic --variant fuzzy --metrics-out results/fuzzy_metrics.json
```

**Process**:

1. Load engineered train/test splits
2. Fit fuzzy cutpoints on training set
3. Generate fuzzy memberships (train + test)
4. Build fuzzy rules (train + test)
5. Concatenate: [28 engineered] + [21 memberships] + [10 rules]
6. Train LightGBM (no monotonic constraints)
7. Evaluate + SHAP explanations

### Hyperparameters

```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    # monotone_constraints NOT applied
)
```

### Performance

**Test Set Results**:

- **ROC-AUC**: 0.7686 (+0.6% vs Baseline Engineered)
- **PR-AUC**: 0.5391 (+5.1% vs Baseline Engineered)
- **Brier Score**: 0.1696 (−0.6%)
- **KS Statistic**: 0.4117 (+2.6%)

**Interpretation**:

- Fuzzy features capture non-linear risk interactions
- Domain-expert rules encode creditworthiness heuristics
- Significant PR-AUC improvement critical for imbalanced data

## 4. Fuzzy-Monotonic Model (Recommended)

### Description

LightGBM trained on **59 features** (same as Fuzzy) with **7 monotonic constraints** ensuring economic consistency.

### Location

`src/models/fuzzy_monotonic.py` (default mode)

### Monotonic Constraints

Enforces directional relationships between features and default probability:

```python
monotone_map = {
    'LIMIT_BAL': -1,              # Higher credit limit → LOWER risk
    'AGE': -1,                    # Older age → LOWER risk
    'PAY_0': +1,                  # Recent late payment → HIGHER risk
    'utilization': +1,            # Higher utilization → HIGHER risk
    'repay_ratio1': -1,           # Higher repayment ratio → LOWER risk
    'delinquency_intensity': +1,  # More delinquency → HIGHER risk
    'paytrend': -1,               # Improving trend → LOWER risk
}
```

**Economic Rationale**:

| Feature                 | Constraint | Justification                                      |
| ----------------------- | ---------- | -------------------------------------------------- |
| `LIMIT_BAL`             | −1         | Higher creditworthiness grants higher limits       |
| `AGE`                   | −1         | Older applicants typically more financially stable |
| `PAY_0`                 | +1         | Recent late payment signals current distress       |
| `utilization`           | +1         | Maxing out credit indicates financial stress       |
| `repay_ratio1`          | −1         | Paying more signals ability to repay               |
| `delinquency_intensity` | +1         | Historical delinquency predicts future default     |
| `paytrend`              | −1         | Improving payment behavior reduces risk            |

### Training

**Command**:

```bash
python src/models/fuzzy_monotonic.py --variant fuzzy_monotonic --metrics-out results/fuzzy_monotonic_metrics.json
```

**Process**:

1. Same fuzzy feature generation as Fuzzy Model
2. Construct monotonic constraint vector
3. Train LightGBM with `monotone_constraints` parameter
4. Validate monotonicity holds
5. Evaluate + SHAP explanations

### Hyperparameters

```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    monotone_constraints=constraint_vector,  # KEY DIFFERENCE
    random_state=42
)
```

**Constraint Vector Construction**:

```python
constraint_vector = [
    monotone_map.get(col, 0)  # 0 = unconstrained
    for col in X_train.columns
]
# Example: [-1, 0, 0, 0, -1, +1, +1, -1, ..., +1, -1, 0, ...]
```

### Performance

**Test Set Results**:

- **ROC-AUC**: 0.7700 (+0.1% vs Fuzzy)
- **PR-AUC**: 0.5498 (+2.0% vs Fuzzy)
- **Brier Score**: 0.1696 (=)
- **KS Statistic**: 0.4144 (+0.7%)

**Interpretation**:

- **Best overall performance** across all metrics
- Monotonic constraints act as **regularization**
- Prevents overfitting to spurious correlations
- Ensures **regulatory compliance** (explainable, consistent)

### Monotonicity Validation

**Method**: Partial dependence validation

```python
# For each constrained feature:
# 1. Generate linearly spaced values
# 2. Fix other features at median
# 3. Predict probabilities
# 4. Check monotonicity: diff(probs) matches constraint sign
```

**Expected Outcome**:

- `LIMIT_BAL` ↑ → default_prob ↓ (monotonic decreasing)
- `utilization` ↑ → default_prob ↑ (monotonic increasing)

## Model Comparison Summary

### Ablation Study Results

| Metric  | Baseline Raw | Baseline Eng | Fuzzy  | Fuzzy-Monotonic | Gain   |
| ------- | ------------ | ------------ | ------ | --------------- | ------ |
| ROC-AUC | 0.7496       | 0.7628       | 0.7686 | **0.7700**      | +2.7%  |
| PR-AUC  | 0.4872       | 0.5129       | 0.5391 | **0.5498**      | +12.9% |
| Brier   | 0.1725       | 0.1706       | 0.1696 | **0.1696**      | −1.7%  |
| KS      | 0.3821       | 0.4014       | 0.4117 | **0.4144**      | +8.5%  |

**Key Insights**:

1. **Feature Engineering**: +5.3% PR-AUC gain (Baseline Raw → Eng)
2. **Fuzzy Logic**: +5.1% PR-AUC gain (Baseline Eng → Fuzzy)
3. **Monotonic Constraints**: +2.0% PR-AUC gain (Fuzzy → Fuzzy-Mono)
4. **Cumulative**: +12.9% PR-AUC gain (end-to-end)

### Computational Cost

| Variant      | Training Time | Features | Model Size |
| ------------ | ------------- | -------- | ---------- |
| Baseline Raw | ~5s           | 23       | 1.2 MB     |
| Baseline Eng | ~6s           | 28       | 1.5 MB     |
| Fuzzy        | ~12s          | 59       | 3.2 MB     |
| Fuzzy-Mono   | ~12s          | 59       | 3.2 MB     |

**Notes**:

- Fuzzy feature generation adds ~2s overhead
- Monotonic constraints negligible overhead
- SHAP explanation: ~8s for 500 samples

## Explainability Features

### SHAP (SHapley Additive exPlanations)

**Method**: TreeExplainer for LightGBM

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_sample)
```

**Outputs**:

- **Feature Importance**: Global ranking
- **Waterfall Plot**: Individual prediction breakdown
- **Beeswarm Plot**: Distribution of feature impacts

**Example Output**:

```
Top 5 Global Feature Importances:
1. PAY_0 (0.185)
2. delinquency_intensity (0.142)
3. utilization (0.098)
4. fuzz_rule_high_util_high_delinq (0.076)
5. LIMIT_BAL (0.063)
```

### Fuzzy Rule Interpretability

**Human-Readable Rules**:

```python
# Rule activation for sample:
{
  'fuzz_rule_high_util_high_delinq': 0.85,  # 85% activation
  'fuzz_rule_low_repay_worsening_trend': 0.62,
  'fuzz_rule_high_pay0_low_repay': 0.41,
  ...
}

# Interpretation:
# "This applicant strongly matches the 'high utilization + high delinquency'
#  risk pattern (85% confidence), indicating severe credit stress."
```

### Monotonic Constraint Transparency

**Verification Report**:

```python
for feature, direction in monotone_map.items():
    if direction == +1:
        print(f"{feature}: Increasing values INCREASE default risk")
    elif direction == -1:
        print(f"{feature}: Increasing values DECREASE default risk")
```

**Regulatory Benefit**:

- Model decisions **directionally consistent** with economic theory
- Avoids counterintuitive predictions (e.g., "higher credit limit → higher risk")
- Passes fairness audits more easily

## Hyperparameter Tuning

### Current Configuration

**LightGBM Hyperparameters**:

- `n_estimators=100`: Balances performance vs. overfitting
- `learning_rate=0.05`: Slow learning for better generalization
- `max_depth=7`: Prevents excessive tree depth
- `num_leaves=31`: Complexity control (2^5 - 1)
- `is_unbalance=True`: Auto class weighting for 22% default rate

### Recommended Tuning (Future Work)

**Grid Search Spaces**:

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [10, 20, 30]
}
```

**Cross-Validation**:

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Evaluate each config on 5 folds
```

**Metric**: Optimize for **PR-AUC** (prioritizes precision on minority class)

## Model Persistence

### Saving Models

**Recommended Structure**:

```python
import joblib

# Save model
joblib.dump(model, 'models/fuzzy_monotonic_v1.pkl')

# Save preprocessing artifacts
joblib.dump(scaler, 'models/preprocessor_v1.pkl')
joblib.dump(fuzzy_cutpoints, 'models/fuzzy_cutpoints_v1.json')

# Save metadata
metadata = {
    'version': 'v1.0',
    'train_date': '2024-01-15',
    'metrics': metrics,
    'hyperparameters': model.get_params()
}
joblib.dump(metadata, 'models/metadata_v1.pkl')
```

### Loading for Inference

```python
# Load artifacts
model = joblib.load('models/fuzzy_monotonic_v1.pkl')
scaler = joblib.load('models/preprocessor_v1.pkl')
fuzzy_cutpoints = joblib.load('models/fuzzy_cutpoints_v1.json')

# Preprocess new data
X_new = preprocess(raw_data, scaler, fuzzy_cutpoints)

# Predict
probs = model.predict_proba(X_new)[:, 1]
```

## Deployment Considerations

### API Wrapper Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CreditApplication(BaseModel):
    LIMIT_BAL: float
    AGE: int
    PAY_0: int
    # ... other features

@app.post("/predict")
def predict(application: CreditApplication):
    # Preprocess
    X = engineer_features(application.dict())
    X_fuzzy = generate_fuzzy_features(X, fuzzy_cutpoints)

    # Predict
    prob = model.predict_proba([X_fuzzy])[0, 1]

    # Explain
    shap_values = explainer([X_fuzzy])

    return {
        "default_probability": prob,
        "risk_category": "High" if prob > 0.5 else "Low",
        "explanations": format_shap(shap_values)
    }
```

### Monitoring

**Metrics to Track**:

- **Data Drift**: KL divergence on feature distributions
- **Prediction Drift**: Distribution shift in output probabilities
- **Performance Degradation**: PR-AUC on recent data
- **Monotonicity Violations**: Validate constraints hold

**Tools**:

- Evidently AI for drift detection
- MLflow for experiment tracking
- Weights & Biases for model monitoring

## Future Enhancements

1. **Deep Learning**: Neural network with monotonic layers (e.g., Monotonic Networks)
2. **Ensemble**: Combine Fuzzy-Monotonic with XGBoost, CatBoost
3. **Causal Inference**: Counterfactual explanations for "what if" scenarios
4. **Fairness Constraints**: Add demographic parity constraints
5. **Online Learning**: Incremental updates as new data arrives
