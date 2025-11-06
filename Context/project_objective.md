X-FuzzyScore: Project Objective & Agent Direction
Primary Research Goal

To build a fuzzy-augmented, monotonic gradient boosting model that predicts credit default and is explainable, calibrated, and robust under class imbalance.

This framework should produce decisions that behave like a domain expert (fuzzy rules), follow economic logic (monotonic constraints), and remain scientifically valid through modern ML evaluation practices (PR-AUC, calibration, SHAP).

Core Modeling Principles for the Agent
Principle	Why
Use tree-based gradient boosting (LightGBM core)	Best for tabular credit datasets
Integrate fuzzy rule features	To embed expert domain logic into model
Apply monotonic constraints	To enforce economically correct behavior
Prefer PR-AUC & Brier over Accuracy	Because recall & calibration > raw accuracy
Always avoid leakage & fit transforms only on train	Regulatory alignment
Explain every prediction through SHAP + fuzzy rule activations	Human interpretability is mandatory
Dataset Roles
Dataset	Purpose
Taiwan Credit Default (main dataset)	Primary benchmark for modeling, calibration, comparisons
German Credit (small dataset)	Interpretability demonstration + fuzzy rule clarity
Mandatory Outputs

results/metrics_baseline.json

results/metrics_fuzzy.json

PR Curve + Calibration Plot

SHAP Summary Plot

German dataset fuzzy rule activation examples

Streamlit dashboard screenshot

Final paper PDF

Minimum Performance Expectation

Fuzzy + Monotonic model should show at least +0.02 PR-AUC lift vs non-fuzzy baseline
AND improved calibration (lower Brier).

Small improvement is acceptable — the innovation is transparency + monotonic + fuzzy augmentation.

Project One-Line Thesis

X-FuzzyScore combines fuzzy reasoning with monotonic gradient boosting for explainable, calibrated credit-risk prediction under class imbalance.