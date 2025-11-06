ML KNOWLEDGE BASE FOR X-FuzzyScore
Purpose

This document captures the core machine learning domain knowledge extracted from prior literature on credit default prediction.
This knowledge guides model selection, evaluation, and interpretability decisions across the X-FuzzyScore framework.

1) Best performing model class for credit default

Tree based ensembles consistently outperform SVM, ANN, Naive Bayes and single classical ML methods on Taiwan credit card default dataset and similar consumer finance datasets.

Examples from literature show Extra Trees, Random Forests, Gradient Boosted Trees perform strongest.

Therefore: LightGBM / Gradient Boosted Trees are the primary modeling backbone in X-FuzzyScore.

2) Metric priority (bank domain reality)

Accuracy is a misleading metric for credit default.

In credit lending:
false negative = extremely expensive → missed defaulter
false positive = manageable cost → rejecting low risk customer

Thus we prioritize:

PR-AUC

Recall

Brier Score (calibration)
not just accuracy.

3) Why fuzzy logic belongs here

Financial domain experts already reason in fuzzy linguistic form:

“limit high”

“recent delays high”

“payment regular, stable”

“debt ratio high = danger”

This maps naturally to fuzzy rule definitions and makes model explainability directly domain consistent vs post-hoc only explainability.

4) Offline + Online knowledge blending

Prior literature splits concepts into two knowledge types:

Type	Meaning
Offline knowledge	long term domain understanding (expert rules, macro behaviour)
Online knowledge	recent behaviour signals, rapidly changing risk

X-FuzzyScore models this by adding fuzzy rule features (offline prior logic) on top of temporal ML features (online signal).

5) Monotonic Constraints Motivation

Economic logic must not be violated:

More delay should never reduce predicted risk

Higher income should never increase risk

Higher credit limit should generally reduce risk

Monotonic GBM constraints enforce economically meaningful structure → this increases fairness, trust, regulatory acceptability.

6) Why Fuzzy + Monotonic + SHAP is novel

Most previous work is either:

pure heuristic rules

OR pure ML black box

OR SHAP/Explainability after modeling

X-FuzzyScore combines all three inside the modeling pipeline:

fuzzy domain priors (interpretable rule activation features)

monotonic constraints (regulatory safe behaviour)

SHAP to explain final feature influence

7) Summary Positioning Sentence

X-FuzzyScore is positioned as a fuzzy-augmented monotonic gradient boosting approach for calibrated and explainable credit risk modeling under class imbalance conditions.