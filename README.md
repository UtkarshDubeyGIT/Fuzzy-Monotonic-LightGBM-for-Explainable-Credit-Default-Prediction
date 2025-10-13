# X-FuzzyScore: Explainable Fuzzy Credit Risk Prediction Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](Research%20paper.pdf)

## 📄 Abstract

This repository presents **X-FuzzyScore**, a novel credit risk prediction framework that integrates fuzzy logic systems with state-of-the-art machine learning models to achieve both high predictive performance and human-interpretable explanations. The framework addresses the critical challenge in financial AI: balancing model accuracy with regulatory compliance and stakeholder trust through transparent, auditable decision-making.

**Key Features:**

- **Dual-layer interpretability**: Linguistic fuzzy rules + SHAP post-hoc explanations
- **Hybrid architecture**: Feature augmentation combining fuzzy risk scores with ML classifiers
- **Strong performance**: ROC-AUC ~0.77 with improved recall for default detection
- **Regulatory-ready**: Auditable fuzzy inference + rigorous feature attribution

📖 **[Read the full research paper](Research%20paper.pdf)**

---

## 🎯 Motivation

Credit risk assessment is a high-stakes domain where model decisions directly impact:

- **Financial institutions**: Loan approval, interest rate determination, portfolio management
- **Consumers**: Access to credit, fair lending practices
- **Regulators**: Compliance with fair lending laws (ECOA, FCRA) and model transparency requirements

Traditional black-box ML models (XGBoost, neural networks) achieve high accuracy but lack interpretability, creating barriers to:

- Regulatory approval and audit
- Stakeholder trust and adoption
- Identification of bias and fairness issues

**X-FuzzyScore** bridges this gap by combining:

1. **Fuzzy logic**: Human-understandable linguistic rules encoding domain knowledge
2. **Machine learning**: Data-driven pattern recognition for complex relationships
3. **Explainable AI**: SHAP values for rigorous feature attribution and validation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Preprocessing                          │
│  • Winsorization (1st-99th percentile outlier handling)         │
│  • Categorical encoding & normalization                          │
│  • Feature engineering (bill ratios, payment aggregates)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼──────────┐       ┌──────────▼───────────┐
│   Fuzzy Inference  │       │  ML Feature Set      │
│      Layer         │       │  (23 baseline        │
│                    │       │   features)          │
│  • Linguistic vars │       │                      │
│  • 8 domain rules  │       └──────────┬───────────┘
│  • Mamdani system  │                  │
│  • Defuzzification │                  │
│                    │                  │
│  Output:           │                  │
│  fuzzy_risk_score  │                  │
└─────────┬──────────┘                  │
          │                             │
          └──────────────┬──────────────┘
                         │
                ┌────────▼─────────┐
                │  Hybrid Fusion   │
                │  (24 features)   │
                └────────┬─────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼──────────┐       ┌──────────▼───────────┐
│  ML Classifiers    │       │  Explainability      │
│  • Logistic Reg    │       │  (SHAP)              │
│  • Random Forest   │       │                      │
│  • XGBoost         │       │  • Global importance │
│  • LightGBM        │       │  • Local attribution │
└─────────┬──────────┘       └──────────┬───────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
                ┌────────▼─────────┐
                │  Risk Prediction │
                │  & Explanation   │
                └──────────────────┘
```

---

## 📊 Datasets

The framework is evaluated on benchmark credit risk datasets:

| Dataset                        | Source                                                                                      | Samples | Features | Class Distribution             | Use Case                   |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ------- | -------- | ------------------------------ | -------------------------- |
| **German Credit**              | [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)     | 1,000   | 20       | 70% / 30% (good/bad)           | Interpretability benchmark |
| **Taiwan Credit Card Default** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000  | 23       | 78% / 22% (no default/default) | Large-scale validation     |

### Data Preprocessing

- **Outlier handling**: Winsorization at 1st and 99th percentiles (reduces skewness from ~5.0 to ~0.07)
- **Categorical encoding**: Consolidation of sparse/undefined categories for stability
- **Feature engineering**: Bill-to-limit ratios, payment status aggregates
- **Train/test split**: 80/20 stratified split (n_train=23,971, n_test=5,993)

---

## 🧮 Methodology

### Fuzzy Inference System

- **Input variables**: Credit limit, payment status, bill ratio, age
- **Membership functions**: Triangular functions with quantile-based parameters
- **Rule base**: 8 domain-knowledge rules (e.g., "IF credit_limit is low AND payment_status is poor THEN risk is high")
- **Inference**: Mamdani system with min t-norm, max s-norm, centroid defuzzification
- **Output**: Fuzzy risk score ∈ [0, 1]

### Machine Learning Models

1. **Logistic Regression**: Linear baseline with L2 regularization
2. **Random Forest**: 100 trees, max depth 10, class weighting
3. **XGBoost**: Gradient boosting with regularization
4. **LightGBM**: Leaf-wise growth, GOSS sampling

### Hybridization Strategies

- **Feature augmentation**: Append fuzzy_risk_score to baseline features (23 → 24 features)
- **Late fusion**: Weighted combination of ML probabilities and fuzzy scores (optional)

### Explainability (SHAP)

- **TreeSHAP**: Efficient Shapley value computation for tree-based models
- **Global importance**: Mean |SHAP| values across samples
- **Local explanations**: Force plots and waterfall diagrams for individual predictions
- **Validation**: Quantifies fuzzy layer contribution vs. raw features

---

## 📈 Results

### Performance Metrics (Test Set)

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC   |
| -------------------- | -------- | --------- | ------ | -------- | --------- |
| Logistic Regression  | 0.816    | 0.693     | 0.432  | 0.533    | 0.751     |
| Random Forest        | 0.819    | 0.677     | 0.505  | 0.578    | 0.767     |
| XGBoost              | 0.817    | 0.682     | 0.488  | 0.568    | 0.766     |
| LightGBM             | 0.817    | 0.678     | 0.499  | 0.575    | 0.767     |
| **LR + Fuzzy**       | 0.816    | 0.693     | 0.436  | 0.535    | 0.752     |
| **RF + Fuzzy**       | 0.818    | 0.675     | 0.505  | 0.578    | 0.769     |
| **XGBoost + Fuzzy**  | 0.818    | 0.681     | 0.496  | 0.574    | 0.769     |
| **LightGBM + Fuzzy** | 0.818    | 0.680     | 0.506  | 0.580    | **0.770** |

**Key Findings:**

- ✅ **Best performance**: LightGBM + Fuzzy (ROC-AUC: 0.770)
- ✅ **Improved recall**: Fuzzy integration boosts sensitivity to defaults
- ✅ **Maintained accuracy**: No significant performance sacrifice for interpretability
- ✅ **Dual explanations**: Fuzzy rules + SHAP values provide complementary insights

---

## 🔍 Interpretability Features

### Layer 1: Fuzzy Rules (Human-Interpretable)

```
R1: IF credit_limit is Low  AND payment_status is Poor → risk is High
R2: IF credit_limit is Low  AND bill_ratio is High     → risk is High
R3: IF payment_status is Poor AND bill_ratio is High   → risk is High
R4: IF credit_limit is High AND payment_status is Good → risk is Low
R5: IF credit_limit is Medium AND payment_status is Fair → risk is Medium
R6: IF age is Young AND payment_status is Poor         → risk is High
R7: IF age is Senior AND payment_status is Good        → risk is Low
R8: IF bill_ratio is Low AND payment_status is Good    → risk is Low
```

### Layer 2: SHAP Post-Hoc Explanations

- **Top predictive features**: Payment status, credit limit, bill amounts
- **Feature interactions**: SHAP captures non-linear relationships missed by fuzzy rules
- **Validation**: Quantifies fuzzy_risk_score importance in hybrid models

### Visual Outputs

- Fuzzy membership functions (credit limit, payment status, bill ratio, age, risk score)
- Fuzzy risk score distribution (defaulters vs. non-defaulters)
- SHAP summary plots (global importance + beeswarm)
- Feature importance rankings

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework.git
cd Credit-Risk-Analysis-and-Prediction-Framework

# Install required packages
pip install -r requirements.txt
```

### Required Libraries

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scikit-fuzzy>=0.4.2
xgboost>=1.5.0
lightgbm>=3.3.0
shap>=0.40.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Usage

#### 1. Exploratory Data Analysis

```bash
jupyter notebook EDA/Draft\ 1\ Germany\ Dataset.ipynb
```

#### 2. Run X-FuzzyScore Pipeline

```python
# Import framework components
from fuzzy_inference import FuzzyRiskSystem
from ml_models import HybridClassifier
from explainability import SHAPExplainer

# Initialize fuzzy system
fuzzy_system = FuzzyRiskSystem()
fuzzy_scores = fuzzy_system.compute_risk_scores(X_train)

# Train hybrid model
hybrid_model = HybridClassifier(model_type='lightgbm')
hybrid_model.fit(X_train, fuzzy_scores, y_train)

# Generate explanations
explainer = SHAPExplainer(hybrid_model)
shap_values = explainer.explain(X_test)
```

#### 3. Generate Results Tables

```bash
python scripts/generate_results_latex.py
```

---

## 📂 Repository Structure

```
Credit-Risk-Analysis-and-Prediction-Framework/
├── EDA/
│   ├── Draft 1 Germany Dataset.ipynb    # Main analysis notebook
│   ├── dataset_explanation.ipynb         # Dataset documentation
│   └── x_fuzzyscore_models/              # Trained models & metrics
│       ├── fuzzy_risk_scores.csv
│       ├── model_comparison.csv
│       └── performance_metrics.json
├── Latex/
│   ├── intro_section.tex                 # Paper introduction
│   ├── methodology_section.tex           # Detailed methodology
│   ├── data_and_exploratory_analysis.tex # Data description & EDA
│   ├── results_section.tex               # Results & evaluation
│   ├── main.tex                          # Main LaTeX document
│   └── figures/                          # Fuzzy & SHAP visualizations
│       ├── germany_008.png (Credit limit membership)
│       ├── germany_009.png (Payment status membership)
│       ├── germany_014.png (Fuzzy score distribution)
│       └── germany_015.png (SHAP summary)
├── scripts/
│   ├── extract_notebook_figures.py       # Extract figures from notebooks
│   └── generate_results_latex.py         # Auto-generate results tables
├── Research paper.pdf                    # Full academic paper
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 📖 Research Paper

The complete research paper is available in this repository:

**[Download Research Paper (PDF)](Research%20paper.pdf)**

### Paper Contents

1. **Introduction**: Motivation, research gap, objectives
2. **Literature Review**: Credit risk modeling, fuzzy logic in finance, explainable AI
3. **Methodology**: X-FuzzyScore architecture, fuzzy inference system, ML models, SHAP integration
4. **Data & Exploratory Analysis**: Dataset overview, preprocessing, feature engineering
5. **Results**: Performance metrics, model comparisons, interpretability evaluation
6. **Discussion**: Insights, limitations, regulatory implications
7. **Conclusion**: Contributions, future work, broader impact

### Citing This Work

If you use this framework in your research, please cite:

```bibtex
@article{xfuzzyscore2025,
  title={X-FuzzyScore: An Explainable Fuzzy Logic Framework for Credit Risk Prediction},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2025},
  url={https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework}
}
```

---

## 🎓 Applications

### Financial Institutions

- **Loan approval systems**: Transparent risk assessment with regulatory-compliant explanations
- **Interest rate determination**: Risk-based pricing with auditable justifications
- **Portfolio management**: Identify high-risk segments with interpretable profiles

### Regulatory Compliance

- **Model validation**: Dual-layer explanations satisfy both technical and business auditors
- **Fair lending**: Identify and mitigate bias through feature attribution
- **Documentation**: Fuzzy rules provide natural-language model documentation

### Research & Education

- **Benchmark framework**: Standard implementation for fuzzy-ML hybrid systems
- **Explainability case study**: Demonstrates complementary interpretability methods
- **Educational tool**: Visual fuzzy inference for teaching credit risk concepts

---

## 🔬 Future Work

- [ ] **Multi-dataset validation**: Test on LendingClub, Kaggle Give Me Some Credit datasets
- [ ] **Fairness analysis**: Integrate demographic parity and equalized odds constraints
- [ ] **Dynamic rule learning**: Automatically discover fuzzy rules from data
- [ ] **Deep learning integration**: Extend framework to neural network models
- [ ] **Real-time deployment**: API service for production credit scoring
- [ ] **Interactive dashboard**: Streamlit/Dash web interface for stakeholders

---

## 👥 Contributors

This project was developed as part of academic research on explainable AI for financial applications.

- **Research Lead**: [Name]
- **Data Engineering**: [Name]
- **ML Development**: [Name]
- **Explainability**: [Name]
- **Paper Writing**: [Name]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the German Credit and Taiwan Credit Card Default datasets
- **Lundberg & Lee (2017)** for the SHAP framework
- **Mamdani (1975)** for foundational work on fuzzy inference systems
- **Open-source community** for scikit-fuzzy, XGBoost, LightGBM, and SHAP libraries

---

## 📧 Contact

For questions, collaborations, or feedback:

- **GitHub Issues**: [Open an issue](https://github.com/UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework/issues)
- **Email**: [utkarsh.dubey.ug23@nsut.ac.in]
- **Paper**: [Research paper.pdf](Research%20paper.pdf)

---

## 🌟 Star History

If you find this work useful, please consider starring the repository and citing our paper!

[![Star History Chart](https://api.star-history.com/svg?repos=UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework&type=Date)](https://star-history.com/#UtkarshDubeyGIT/Credit-Risk-Analysis-and-Prediction-Framework&Date)

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Status**: Research Complete | Paper Submitted
