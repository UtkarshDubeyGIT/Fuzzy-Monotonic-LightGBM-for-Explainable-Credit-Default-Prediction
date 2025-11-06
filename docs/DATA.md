# Data Documentation

## Datasets Overview

This project uses two primary datasets for credit risk modeling:

| Dataset                    | Samples | Features | Default Rate | Use Case                       |
| -------------------------- | ------- | -------- | ------------ | ------------------------------ |
| Taiwan Credit Card Default | 30,000  | 23       | 22.1%        | Model training & evaluation    |
| German Credit              | 1,000   | 20       | 30.0%        | Interpretability demonstration |

## Taiwan Credit Card Default Dataset

### Source

- **Origin**: UCI Machine Learning Repository
- **Reference**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. _Expert Systems with Applications_, 36(2), 2473-2480.
- **Collection Period**: April 2005 - September 2005
- **Location**: Taiwan

### Schema

#### Raw Features (23 total)

**Target Variable**:

- `default.payment.next.month`: Binary indicator (1 = default, 0 = non-default)

**Demographic Features** (4):

- `LIMIT_BAL` (int): Credit limit in New Taiwan dollars
- `SEX` (int): 1 = male, 2 = female
- `EDUCATION` (int): 1 = graduate school, 2 = university, 3 = high school, 4 = others, 5/6 = unknown
- `MARRIAGE` (int): 1 = married, 2 = single, 3 = others

**Payment History** (6 months):

- `PAY_0` to `PAY_6` (int): Repayment status from September (PAY_0) to April (PAY_6)
  - -1 = pay duly
  - 1 = payment delay for 1 month
  - 2 = payment delay for 2 months
  - ...
  - 8 = payment delay for 8 months
  - 9 = payment delay for 9+ months

**Bill Amounts** (6 months):

- `BILL_AMT1` to `BILL_AMT6` (float): Bill statement amount from September (1) to April (6) in NT dollars

**Payment Amounts** (6 months):

- `PAY_AMT1` to `PAY_AMT6` (float): Previous payment from September (1) to April (6) in NT dollars

**Age**:

- `AGE` (int): Age in years

### Engineered Features

The preprocessing pipeline generates 5 additional features when `--engineered` flag is used:

#### 1. BILL_AMT_AVG

```python
BILL_AMT_AVG = (BILL_AMT1 + BILL_AMT2 + ... + BILL_AMT6) / 6
```

**Rationale**: Smooths temporal variance, captures average debt level

#### 2. utilization

```python
utilization = BILL_AMT1 / LIMIT_BAL
```

**Rationale**: Credit utilization ratio, key indicator of financial stress  
**Range**: [0, ∞) (can exceed 1.0 if over-limit)

#### 3. repay_ratio1

```python
repay_ratio1 = PAY_AMT1 / (BILL_AMT1 + 1e-6)
```

**Rationale**: Recent repayment behavior (most recent month)  
**Range**: [0, ∞)  
**Note**: Small epsilon prevents division by zero

#### 4. delinquency_intensity

```python
delinquency_intensity = sum([PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6])
```

**Rationale**: Cumulative severity of payment delays  
**Range**: [-6, 54] (theoretical)

#### 5. paytrend

```python
paytrend = PAY_0 - PAY_6
```

**Rationale**: Payment behavior trend (improving vs. worsening)  
**Interpretation**:

- Negative value = improving (was worse 6 months ago)
- Positive value = worsening (was better 6 months ago)
- Zero = stable

### Data Quality

**Missing Values**: None (complete dataset)

**Class Imbalance**:

- Default (1): 6,636 samples (22.12%)
- Non-default (0): 23,364 samples (77.88%)
- **Strategy**: `is_unbalance=True` in LightGBM handles imbalance

**Outliers**:

- `LIMIT_BAL`: Range [10,000 - 1,000,000], some extreme values
- `AGE`: Range [21 - 79], plausible distribution
- Bill amounts: Can be negative (credit balance), require robust scaling

**Data Types**:

- 18 integer features (categorical treated as numeric)
- 5 float features (monetary amounts)

### Exploratory Data Analysis Insights

#### Univariate Analysis

- **Credit Limit**: Right-skewed distribution, median ~150K NT
- **Age**: Normal-like distribution, mean ~35 years
- **Bill Amounts**: Heavy right tail, many small balances
- **Payment Amounts**: Spike at zero (missed payments)

#### Bivariate Analysis

- **Utilization vs. Default**: Strong positive correlation (0.42)
- **Delinquency Intensity vs. Default**: Strongest predictor (0.49)
- **Repayment Ratio vs. Default**: Negative correlation (-0.31)
- **Age vs. Default**: Weak negative correlation (-0.08)

#### Multicollinearity

- High correlation between consecutive `BILL_AMTx` features (0.85-0.95)
- High correlation between consecutive `PAY_AMTx` features (0.60-0.80)
- **Mitigation**: Tree-based models handle multicollinearity well

### Train/Test Split

**Strategy**: Stratified random split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**Sizes**:

- Training set: 24,000 samples (80%)
- Test set: 6,000 samples (20%)
- Default rate preserved in both splits (~22%)

**Outputs**:

- `data/processed_baseline_raw/train.csv` (24,000 × 23)
- `data/processed_baseline_raw/test.csv` (6,000 × 23)
- `data/processed_baseline_engineered/train.csv` (24,000 × 28)
- `data/processed_baseline_engineered/test.csv` (6,000 × 28)

### Feature Scaling

**Method**: RobustScaler (sklearn)

```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Rationale**:

- Robust to outliers (uses median, IQR)
- Handles heavy-tailed distributions in bill amounts
- Maintains relative distances

**Scaled Features**:

- Numeric features only (not categorical)
- Monetary amounts: `LIMIT_BAL`, `BILL_AMTx`, `PAY_AMTx`, engineered features

### Encoding

**Categorical Features**:

- `SEX`: Binary, kept as-is (1/2)
- `EDUCATION`: Label encoded (1-6)
- `MARRIAGE`: Label encoded (1-3)

**Payment Status**:

- `PAY_0` to `PAY_6`: Ordinal, kept as numeric (-1 to 9)

## German Credit Dataset

### Source

- **Origin**: UCI Machine Learning Repository (Statlog)
- **Reference**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
- **Location**: Germany
- **Samples**: 1,000

### Purpose

Demonstrates **interpretability** on a smaller, publicly-validated dataset:

- Fuzzy rule explanations
- SHAP analysis
- Feature interaction visualization

### Schema (20 features)

**Target**:

- `Creditability`: 1 = good credit, 2 = bad credit (inverted from typical)

**Features** (mixed types):

- Account balance status (categorical)
- Payment status of previous credit (categorical)
- Purpose of credit (categorical)
- Credit amount (numeric)
- Employment duration (categorical)
- Marital status (categorical)
- Guarantors (categorical)
- Residence duration (numeric)
- Assets (categorical)
- Age (numeric)
- Other installment plans (categorical)
- Housing (categorical)
- Number of existing credits (numeric)
- Job (categorical)
- Number of dependents (numeric)
- Telephone (binary)
- Foreign worker (binary)

### Use Case

Located in `src/models/german_interpretability.py`, generates:

- Fuzzy rule activations for individual predictions
- SHAP waterfall plots
- Feature importance rankings
- Used in **EDA** section of LaTeX papers

## Data Preprocessing Pipeline

### Workflow

```
1. Load Raw CSV
   ↓
2. Drop unnecessary columns (e.g., ID)
   ↓
3. Rename columns (standardize)
   ↓
4. Handle missing values (none in Taiwan dataset)
   ↓
5. Feature Engineering (if --engineered)
   ↓
6. Encode categorical variables
   ↓
7. Train/Test split (stratified)
   ↓
8. Scale numeric features (RobustScaler on train, apply to test)
   ↓
9. Save to data/processed_*/
```

### Code Location

**File**: `src/data/preprocess.py`

**Main Function**:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engineered', action='store_true')
    args = parser.parse_args()

    # Load Taiwan dataset
    df = _read_taiwan_csv(DATA_PATH)

    # Optional feature engineering
    if args.engineered:
        df = _add_engineered_features(df)

    # Split
    X_train, X_test, y_train, y_test = _split_data(df)

    # Scale
    X_train, X_test = _apply_scaling(X_train, X_test)

    # Save
    _save_splits(X_train, X_test, y_train, y_test, args.engineered)
```

### Execution

**Raw features only**:

```bash
python src/data/preprocess.py
```

**With engineered features**:

```bash
python src/data/preprocess.py --engineered
```

## Fuzzy Feature Generation

### Methodology

For each numeric base feature, generate three fuzzy memberships:

#### Cutpoint Calculation

```python
def _fit_fuzzy_cutpoints(series: pd.Series) -> Dict[str, float]:
    return {
        'min': series.min(),
        'p25': series.quantile(0.25),
        'p33': series.quantile(0.33),
        'p50': series.quantile(0.50),
        'p67': series.quantile(0.67),
        'p75': series.quantile(0.75),
        'max': series.max()
    }
```

#### Membership Functions

**Low Membership** (Trapezoid):

```python
if x <= p25:
    return 1.0
elif p25 < x < mid:
    return (mid - x) / (mid - p25)
else:
    return 0.0
```

**Medium Membership** (Triangle):

```python
if p25 < x <= p50:
    return (x - p25) / (p50 - p25)
elif p50 < x < p75:
    return (p75 - x) / (p75 - p50)
else:
    return 0.0
```

**High Membership** (Trapezoid):

```python
if x <= mid:
    return 0.0
elif mid < x < p75:
    return (x - mid) / (p75 - mid)
else:
    return 1.0
```

### Fuzzy Base Features

Selected features for fuzzy transformation:

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

**Output**: 7 base features × 3 memberships = **21 fuzzy features**

### Fuzzy Rule Features

Derived from fuzzy memberships using min-AND logic:

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

**Output**: **10 fuzzy rule features**

## Final Feature Space

### Baseline Raw Model

- **Input features**: 23 (original)
- **After preprocessing**: 23 scaled + encoded

### Baseline Engineered Model

- **Input features**: 28 (23 original + 5 engineered)
- **After preprocessing**: 28 scaled + encoded

### Fuzzy Model

- **Input features**: 28 + 21 fuzzy memberships + 10 rules = **59 features**

### Fuzzy-Monotonic Model

- **Input features**: 59 (same as Fuzzy)
- **Monotonic constraints**: 7 features constrained

## Data Storage

### Directory Structure

```
data/
├── processed_baseline_raw/
│   ├── train.csv         # 24,000 × 23
│   └── test.csv          # 6,000 × 23
└── processed_baseline_engineered/
    ├── train.csv         # 24,000 × 28
    └── test.csv          # 6,000 × 28

Data/
├── taiwan_default_of_credit_card_clients.csv  # Raw
├── german_credit.csv                          # Raw
└── german_fuzzy_rules.csv                     # German fuzzy outputs
```

### File Formats

**CSV Structure**:

```csv
LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,...,default
20000,2,2,1,24,-1,...,0
120000,2,2,2,26,-1,...,0
...
```

**Fuzzy CSV Structure** (generated during training):

```csv
LIMIT_BAL,AGE,PAY_0,...,LIMIT_BAL_low,LIMIT_BAL_med,LIMIT_BAL_high,...,default
20000,24,-1,...,0.85,0.15,0.00,...,0
```

## Data Versioning

**Current Version**: v1.0

- Preprocessing: RobustScaler + stratified split
- Feature engineering: 5 features
- Fuzzy bases: 7 features
- Fuzzy rules: 10 rules

**Recommended for Production**:

- Track preprocessing parameters (scaler state, cutpoints)
- Version control `fuzzy_cutpoints.json`
- Log feature engineering code hash
- Monitor data drift (KL divergence on feature distributions)

## Privacy & Ethics

### Compliance

- Taiwan dataset: Anonymized, no PII
- German dataset: Historical, public domain
- GDPR considerations: No personal identifiers

### Fairness

- `SEX` feature: Included but monitored for bias
- `EDUCATION`, `MARRIAGE`: May correlate with protected classes
- **Mitigation**: Explainability via SHAP enables fairness auditing

### Responsible Use

- Model for research and education
- Production deployment requires fairness testing
- Regular bias monitoring recommended
