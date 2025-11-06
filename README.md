# Fuzzy-Monotonic LightGBM for Explainable Credit Default Prediction# X-FuzzyScore: Explainable Fuzzy Credit-Risk Prediction Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Paper](https://img.shields.io/badge/paper-Extended_IEEE-blue)](Latex/extended_ieee.tex)

An explainable, human-interpretable AI system for credit-risk prediction that combines **Fuzzy Logic**, **Machine Learning**, and **Explainability** (SHAP) with an interactive visualization frontend.

> **A hybrid explainable AI framework combining fuzzy linguistic reasoning with monotonic gradient boosting for regulatory-compliant credit default prediction.**

## Project Objectives

## 🎯 Overview

1. **Predict** credit-risk/loan-default probability for individuals or companies

This research project addresses the critical trade-off between **predictive accuracy** and **regulatory interpretability** in financial credit risk modeling. We propose a novel **Fuzzy-Monotonic LightGBM** framework that achieves competitive performance (ROC-AUC ~0.77) while maintaining structural transparency through:2. **Interpret** every prediction in human language and visuals

3. **Integrate** fuzzy reasoning ("high income", "medium debt") with ML accuracy

- 🧩 **Fuzzy Membership Functions**: Human-interpretable linguistic variables (Low/Medium/High)4. **Visualize** model results, fuzzy rules, and SHAP explanations via web dashboard

- 📊 **Monotonic Constraints**: Economic priors enforced in gradient boosting5. **Publish** results as an academic research paper

- 🔧 **Behavioral Feature Engineering**: Domain-driven credit indicators

- 🔍 **Multi-Layer Explainability**: Structural (fuzzy + monotonic) + attributional (SHAP)## 📊 Datasets

### Key Results| Dataset | Source | Size | Role | Key Features |

|---------|--------|------|------|--------------|

| Metric | Baseline Raw | Baseline + Engineered | Fuzzy | **Fuzzy-Monotonic** || **Taiwan Credit Card Default** | UCI ML Repository | 30,000 samples | Primary modeling & ablation | Temporal repayment behavior (6 months), demographic attributes, billing/payment amounts |

|--------|--------------|----------------------|-------|---------------------|| **German Credit** | UCI Statlog | 1,000 samples | Interpretability demonstration | Categorical financial stability indicators, loan characteristics |

| **ROC-AUC** | 0.7744 | 0.7733 | 0.7701 | **0.7700** |

| **PR-AUC** | 0.5477 | 0.5496 | 0.5485 | **0.5498** ↑ |### Dataset Characteristics

| **Brier Score** | 0.1725 | 0.1730 | 0.1687 | **0.1696** ↓ |

| **KS Statistic** | 0.4235 | 0.4231 | 0.4208 | **0.4144** |**Taiwan Dataset**:

- Target: Next-month default (binary)

✅ **Best PR-AUC** for minority class detection - Features: PAY_0..PAY_6 (repayment status), BILL_AMT1..6, PAY_AMT1..6, demographics

✅ **Improved calibration** for probability estimates - Class imbalance: ~22% default rate

✅ **Economic consistency** via monotonic constraints - Strong predictive signals: Repayment history (IV > 0.5)

✅ **Regulatory alignment** (Basel II/III, IFRS-9, ECB TRIM)

**German Dataset**:

---- Target: Good/Bad credit risk (binary)

- Features: Checking/saving accounts, housing, purpose, age, credit amount, duration

## 📁 Project Structure- Moderate imbalance with interpretable categorical features

- Used for fuzzy rule transparency demonstration

````

├── Data/                           # Raw datasets## 🏗️ Architecture

│   ├── german_credit.csv          # German Credit (Statlog) - 1,000 samples

│   └── taiwan_default_of_credit_card_clients.csv  # Taiwan Credit - 30,000 samples```

├── data/                          # Processed data┌─────────────────────────────────────────────────────────────┐

│   ├── processed_baseline_raw/    # Baseline without engineering│                    Data Preprocessing                        │

│   └── processed_baseline_engineered/  # With engineered features│  • Feature alignment & normalization (RobustScaler)          │

├── src/│  • Categorical encoding (LabelEncoder)                       │

│   ├── data/│  • Train/test stratified split (80/20)                       │

│   │   └── preprocess.py          # Feature engineering pipeline└────────────────────┬────────────────────────────────────────┘

│   └── models/                     │

│       ├── baseline.py            # Logistic Regression & LightGBM baselines┌────────────────────▼────────────────────────────────────────┐

│       ├── fuzzy_monotonic.py     # Fuzzy-Monotonic LightGBM (main model)│            Behavioral Feature Engineering                    │

│       ├── run_ablation.py        # Ablation study orchestrator│  • BILL_AMT_AVG: Mean monthly bill statements                │

│       └── plot_ablation_pr_auc.py # Visualization│  • Utilization: Bill amount / credit limit                   │

├── Latex/                         # Research papers (LaTeX)│  • Delinquency intensity: Max payment delay                  │

│   ├── extended_ieee.tex          # 📄 Extended IEEE paper with EDA│  • Payment trend: Slope of repayment changes                 │

│   ├── ieee_conference.tex        # IEEE conference format└────────────────────┬────────────────────────────────────────┘

│   ├── eda.tex                    # Exploratory Data Analysis report                     │

│   └── elsevier_format.tex        # Elsevier journal format┌────────────────────▼────────────────────────────────────────┐

├── results/                       # Model outputs & metrics│              Fuzzy Membership Layer                          │

│   ├── ablation_table.json        # Ablation study results│  • Linguistic variables: Low / Medium / High                 │

│   ├── *_metrics.json             # Per-variant metrics│  • Percentile-based cut-points (training data)               │

│   ├── pr_curve.png               # Precision-Recall curves│  • Rule activations: min(AND) operators                      │

│   ├── calibration.png            # Calibration curves│  • Human-readable semantics for risk drivers                 │

│   └── shap_fuzzy.png             # SHAP summary plot└────────────────────┬────────────────────────────────────────┘

└── docs/                          # Documentation (see below)                     │

```┌────────────────────▼────────────────────────────────────────┐

│          Monotonic LightGBM Ensemble                         │

---│  • Gradient boosted decision trees                           │

│  • Monotonic constraints on economic priors                  │

## 🚀 Quick Start│  • Class-balanced training (class_weight='balanced')         │

│  • Calibrated probability outputs                            │

### Prerequisites└────────────────────┬────────────────────────────────────────┘

                     │

- Python 3.8 or higher┌────────────────────▼────────────────────────────────────────┐

- pip or conda package manager│             Explainability Layer                             │

- (Optional) Docker for dev container│  • SHAP: Feature attribution (global + local)                │

│  • Fuzzy rule activations: Structural transparency           │

### Installation│  • Monotonicity: Economic consistency guarantees             │

└────────────────────┬────────────────────────────────────────┘

```bash                     │

# Clone the repository                     ▼

git clone https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework.git        ┌──────────────────────────┐

cd Credit-Risk-Analysis-and-Prediction-Framework        │   Outputs & Evaluation   │

        │  • Default probability   │

# Create virtual environment (recommended)        │  • Risk label            │

python -m venv venv        │  • Feature attributions  │

source venv/bin/activate  # On Windows: venv\Scripts\activate        │  • Rule activations      │

        │  • Calibration curves    │

# Install dependencies        └──────────────────────────┘

pip install pandas numpy scikit-learn lightgbm matplotlib seaborn shap```

````

## ⚙️ Tech Stack

### Running the Pipeline

| Layer | Tools / Libraries |

#### 1. Data Preprocessing| ------------------ | ----------------------------------- |

| **Data** | pandas, numpy, sklearn |

```bash| **Fuzzy Logic**    | scikit-fuzzy                        |

# Process Taiwan dataset with feature engineering| **ML / Ensemble**  | xgboost, lightgbm                   |

python src/data/preprocess.py --engineered| **Explainability** | shap, lime                          |

| **Visualization**  | streamlit, dash, plotly, matplotlib |

# Output: data/processed_baseline_engineered/{train.csv, test.csv}| **Documentation**  | Overleaf/LaTeX, GitHub              |

```

## Quick Start

**Engineered Features:**

- `BILL_AMT_AVG`: Mean of 6-month bill statements### Prerequisites

- `utilization`: Bill amount / credit limit ratio

- `repay_ratio1`: Payment amount / bill ratio (month 1)- Python 3.8+

- `delinquency_intensity`: Max payment delay across 6 months- Docker (optional, for dev container)

- `paytrend`: Slope of payment behavior over time

### Installation

#### 2. Baseline Model Training

````bash

```bash# Clone repository

# Train Logistic Regression + LightGBM baselinesgit clone https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework.git

python src/models/baseline.pycd Credit-Risk-Analysis-and-Prediction-Framework



# Outputs:# Install dependencies

# - results/metrics_baseline.jsonpip install -r requirements.txt

# - results/pr_curve.png

# - results/calibration.png# Run preprocessing

```python src/data/preprocess.py



#### 3. Fuzzy-Monotonic Model# Train model

python src/ml/train.py

```bash

# Train full fuzzy-monotonic model# Launch dashboard

python src/models/fuzzy_monotonic.pystreamlit run src/visualization/dashboard.py

````

# Train fuzzy-only (skip monotonic constraints)

python src/models/fuzzy_monotonic.py --skip-monotonic --variant fuzzy## Evaluation Metrics

# Outputs:| Category | Metrics |

# - results/metrics_fuzzy.json (or specified variant)| -------------------- | ------------------------------------------------- |

# - results/shap_fuzzy.png| **Performance** | Accuracy, Precision, Recall, F1, AUC |

````| **Interpretability** | Rule count, average rule length, SHAP consistency |

| **Usability**        | Expert feedback / human interpretability rating   |

#### 4. Complete Ablation Study| **Visualization**    | Clarity, interaction smoothness                   |



```bash##  Project Timeline

# Run all variants and generate comparison table

python src/models/run_ablation.py| Week | Milestone                                |

| ---- | ---------------------------------------- |

# Outputs:| 1-2  | Literature review, finalize research gap |

# - results/ablation_table.json| 3    | Dataset collection & preprocessing       |

# - Markdown table printed to console| 4-5  | Build fuzzy + ML model, test baseline    |

```| 6    | Integrate SHAP & generate explanations   |

| 7    | Develop visualization dashboard          |

#### 5. Visualization| 8    | Compile results, write & format paper    |



```bash##  Contributing

# Generate PR-AUC comparison plot

python src/models/plot_ablation_pr_auc.pyWe welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.



# Output: results/ablation_pr_auc.png##  Citation

````

If you use this framework in your research, please cite:

---

````bibtex

## 🔬 Methodology@article{xfuzzyscore2024,

  title={X-FuzzyScore: An Explainable Fuzzy Credit-Risk Prediction Framework},

### 1. Behavioral Feature Engineering  author={Your Name and Team},

  journal={TBD},

Domain-grounded features capture stable credit behavior:  year={2024}

}

```python```

# Aggregated spending behavior

BILL_AMT_AVG = mean(BILL_AMT1, ..., BILL_AMT6)### Dataset Citations



# Credit utilization- UCI German Credit Dataset (Statlog)

utilization = BILL_AMT_AVG / LIMIT_BAL  # clipped [0,1]- UCI Default of Credit Card Clients Dataset (Yeh & Lien 2009)

- Kaggle LendingClub Loan Data (wordsforthewise)

# Repayment discipline

repay_ratio1 = PAY_AMT1 / BILL_AMT1  # clipped [0,1]##  License



# Delinquency severityThis project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

delinquency_intensity = max(PAY_0, ..., PAY_6)

##  Links

# Payment trend (behavioral drift)

paytrend = (PAY_AMT6 - PAY_AMT1) / (PAY_AMT1 + ε)  # clipped [-1,1]- **Documentation**: [docs/](docs/)

```- **Research Paper**: [docs/paper/](docs/paper/)

- **Issues**: [GitHub Issues](https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework/issues)

### 2. Fuzzy Membership Layer- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework/discussions)



Linguistic variables defined using percentile-based cutpoints (train-only):---



```**Built with ❤️ for transparent and responsible credit risk assessment**

Low:    [min, 33rd percentile]
Medium: [25th, 75th percentile]  # overlapping transitions
High:   [67th percentile, max]
````

**Example Rule Activation:**

```
IF utilization=High AND delinquency_intensity=High
THEN risk=High (activation = 0.87)
```

### 3. Monotonic Constraints

Economic priors enforced in LightGBM:

| Feature                 | Constraint | Economic Rationale                             |
| ----------------------- | ---------- | ---------------------------------------------- |
| `LIMIT_BAL`             | ↑ → risk ↓ | Higher credit limit indicates creditworthiness |
| `AGE`                   | ↑ → risk ↓ | Older borrowers generally more stable          |
| `PAY_0`                 | ↑ → risk ↑ | Recent late payments increase default risk     |
| `utilization`           | ↑ → risk ↑ | High utilization signals financial stress      |
| `repay_ratio1`          | ↑ → risk ↓ | Higher repayment ratio reduces risk            |
| `delinquency_intensity` | ↑ → risk ↑ | Past delinquencies predict future default      |
| `paytrend`              | ↑ → risk ↓ | Improving payment behavior lowers risk         |

### 4. Explainability

**Structural Transparency:**

- Fuzzy rules provide linguistic justifications
- Monotonic constraints ensure economic consistency

**Attributional Explainability:**

- SHAP (TreeExplainer) for feature importance
- Instance-level and global attribution

---

## 📊 Datasets

### Taiwan Credit Card Default (Primary)

- **Source**: UCI ML Repository (Yeh & Lien, 2009)
- **Samples**: 30,000 credit card clients
- **Target**: Next-month default (binary)
- **Features**:
  - Demographics: age, education, marriage
  - Repayment history: PAY_0 to PAY_6
  - Bills: BILL_AMT1 to BILL_AMT6
  - Payments: PAY_AMT1 to PAY_AMT6
- **Class balance**: 22% default rate
- **Role**: Primary modeling, ablation, evaluation

### German Credit (Interpretability)

- **Source**: UCI Statlog
- **Samples**: 1,000 loan applicants
- **Target**: Good/Bad credit risk
- **Features**: Categorical (checking/saving accounts, housing, purpose) + Numeric (age, credit amount, duration)
- **Role**: Fuzzy rule transparency demonstration

---

## 📈 Evaluation Metrics

| Metric           | Purpose                    | Why It Matters                                                      |
| ---------------- | -------------------------- | ------------------------------------------------------------------- |
| **PR-AUC**       | Precision-recall trade-off | Primary metric for imbalanced datasets; sensitive to minority class |
| **ROC-AUC**      | Discrimination power       | Overall ranking ability across thresholds                           |
| **Brier Score**  | Probability calibration    | Quality of probability estimates for provisioning                   |
| **KS Statistic** | Score separation           | Industry-standard for risk model validation                         |

❌ **Accuracy not used** (misleading for imbalanced data)

---

## 🛠️ Tech Stack

| Component           | Tools                                   |
| ------------------- | --------------------------------------- |
| **Data Processing** | pandas, numpy, scikit-learn             |
| **Modeling**        | LightGBM, scikit-learn                  |
| **Explainability**  | SHAP (TreeExplainer)                    |
| **Visualization**   | matplotlib, seaborn                     |
| **Documentation**   | LaTeX (Overleaf-compatible)             |
| **Development**     | Python 3.8+, VS Code, Docker (optional) |

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: System design and components
- **[Data Documentation](docs/DATA.md)**: Datasets, preprocessing, feature engineering
- **[Model Documentation](docs/MODELS.md)**: Model variants, methodology, hyperparameters
- **[Usage Guide](docs/USAGE.md)**: Detailed examples and workflows
- **[API Reference](docs/API.md)**: Code-level documentation

---

## 📝 Research Papers

All papers are available in `Latex/`:

1. **Extended IEEE Format** (`extended_ieee.tex`): Complete paper with comprehensive EDA
2. **IEEE Conference** (`ieee_conference.tex`): Condensed conference submission
3. **Elsevier Journal** (`elsevier_format.tex`): Journal-style format
4. **EDA Report** (`eda.tex`): Standalone exploratory data analysis

### Compiling Papers

```bash
cd Latex/
pdflatex extended_ieee.tex
bibtex extended_ieee
pdflatex extended_ieee.tex
pdflatex extended_ieee.tex
```

---

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{dubey2024fuzzy,
  title={Fuzzy-Monotonic LightGBM for Explainable Credit Default Prediction},
  author={Dubey, Utkarsh and Singla, Kanav and Bhardwaj, Dushyant},
  booktitle={Proceedings of IEEE Conference},
  year={2024},
  organization={Netaji Subhas University of Technology}
}
```

### Dataset Citations

```bibtex
@misc{yeh2009default,
  title={Default of credit card clients dataset},
  author={Yeh, I-Cheng and Lien, Che-hui},
  year={2009},
  publisher={UCI Machine Learning Repository}
}

@misc{german1994statlog,
  title={Statlog (German Credit Data) Dataset},
  publisher={UCI Machine Learning Repository}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt  # if available

# Run tests
pytest tests/  # if test suite exists

# Format code
black src/
isort src/
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Links & Resources

- **Repository**: [GitHub](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework)
- **Issues**: [Report bugs or request features](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework/issues)
- **Discussions**: [Q&A and community](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework/discussions)

### External References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Basel II/III Guidelines](https://www.bis.org/basel_framework/)
- [IFRS 9 Financial Instruments](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)

---

## 🏆 Acknowledgments

- **Netaji Subhas University of Technology**, Department of Computer Science
- UCI Machine Learning Repository for datasets
- Open-source community (scikit-learn, LightGBM, SHAP)

---

## 🔮 Future Work

- [ ] Automated monotonic prior discovery
- [ ] Uncertainty quantification for probability estimates
- [ ] Macroeconomic stress testing and drift analysis
- [ ] Conversion to compact scorecards for production
- [ ] Multi-dataset cross-portfolio validation
- [ ] Causal structure constraints
- [ ] Real-time streaming model updates

---

**Built with ❤️ for transparent and responsible credit risk assessment**
