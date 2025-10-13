# 🧠 X-FuzzyScore: Explainable Fuzzy Credit-Risk Prediction Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An explainable, human-interpretable AI system for credit-risk prediction that combines **Fuzzy Logic**, **Machine Learning**, and **Explainability** (SHAP) with an interactive visualization frontend.

##  Project Objectives

1. **Predict** credit-risk/loan-default probability for individuals or companies
2. **Interpret** every prediction in human language and visuals
3. **Integrate** fuzzy reasoning ("high income", "medium debt") with ML accuracy
4. **Visualize** model results, fuzzy rules, and SHAP explanations via web dashboard
5. **Publish** results as an academic research paper

##  System Architecture

```
Dataset(s)
   │
   ├── Data Preprocessing
   │       ├─ Feature alignment & normalization
   │       ├─ Categorical encoding
   │       └─ Dataset integration (German + Taiwan + LendingClub)
   │
   ├── Fuzzy Layer
   │       ├─ Define linguistic variables (Low/Med/High)
   │       └─ Apply fuzzy rules
   │
   ├── ML Ensemble Layer
   │       └─ XGBoost / LightGBM model for prediction
   │
   ├── Explainability Layer
   │       └─ SHAP / LIME for feature attribution
   │
   ├── Visualization Frontend
   │       ├─ Dashboard (Streamlit/Dash)
   │       ├─ Risk gauge, SHAP bar plots
   │       └─ Fuzzy rule activation viewer
   │
   └── Outputs → Probability, Risk Label, Explanations, Visuals
```

##  Datasets

| Dataset                    | Source            | Size           | Use                         |
| -------------------------- | ----------------- | -------------- | --------------------------- |
| German Credit              | UCI ML Repository | ~1,000 samples | Small interpretable dataset |
| Taiwan Credit Card Default | UCI ML Repository | 30,000 samples | Large-scale testing         |
| LendingClub Loan Data      | Kaggle            | 100k+          | Real-world validation       |

##  Expected Outputs

| Type                 | Example                                                          |
| -------------------- | ---------------------------------------------------------------- |
| **Probability**      | 0.87 → 87% chance of repayment                                   |
| **Risk Label**       | "Low Risk", "Medium Risk", "High Risk"                           |
| **Fuzzy Rules**      | "IF income = high AND debt = low → risk = low (activation 0.82)" |
| **SHAP Explanation** | income -0.18 → reduced risk; debt +0.07 → increased risk         |
| **Visualization**    | Dashboard with gauge, SHAP bars, fuzzy memberships               |

## ⚙️ Tech Stack

| Layer              | Tools / Libraries                   |
| ------------------ | ----------------------------------- |
| **Data**           | pandas, numpy, sklearn              |
| **Fuzzy Logic**    | scikit-fuzzy                        |
| **ML / Ensemble**  | xgboost, lightgbm                   |
| **Explainability** | shap, lime                          |
| **Visualization**  | streamlit, dash, plotly, matplotlib |
| **Documentation**  | Overleaf/LaTeX, GitHub              |

##  Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional, for dev container)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework.git
cd Credit-Risk-Analysis-and-Prediction-Framework

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/data/preprocess.py

# Train model
python src/ml/train.py

# Launch dashboard
streamlit run src/visualization/dashboard.py
```

##  Evaluation Metrics

| Category             | Metrics                                           |
| -------------------- | ------------------------------------------------- |
| **Performance**      | Accuracy, Precision, Recall, F1, AUC              |
| **Interpretability** | Rule count, average rule length, SHAP consistency |
| **Usability**        | Expert feedback / human interpretability rating   |
| **Visualization**    | Clarity, interaction smoothness                   |

##  Project Timeline

| Week | Milestone                                |
| ---- | ---------------------------------------- |
| 1-2  | Literature review, finalize research gap |
| 3    | Dataset collection & preprocessing       |
| 4-5  | Build fuzzy + ML model, test baseline    |
| 6    | Integrate SHAP & generate explanations   |
| 7    | Develop visualization dashboard          |
| 8    | Compile results, write & format paper    |

##  Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

##  Citation

If you use this framework in your research, please cite:

```bibtex
@article{xfuzzyscore2024,
  title={X-FuzzyScore: An Explainable Fuzzy Credit-Risk Prediction Framework},
  author={Your Name and Team},
  journal={TBD},
  year={2024}
}
```

### Dataset Citations

- UCI German Credit Dataset (Statlog)
- UCI Default of Credit Card Clients Dataset (Yeh & Lien 2009)
- Kaggle LendingClub Loan Data (wordsforthewise)

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

##  Links

- **Documentation**: [docs/](docs/)
- **Research Paper**: [docs/paper/](docs/paper/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework/discussions)

---

**Built with ❤️ for transparent and responsible credit risk assessment**
