from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

def load_german():
    root = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(root / "Data" / "german_credit.csv")

    # standardize target  good->0 bad->1
    df["Risk"] = df["Risk"].map({"good":0,"bad":1}).astype(int)

    return df

def fuzzy_membership_numeric(series: pd.Series):
    """simple 3 region fuzzy split based on robust percentiles"""
    q1 = np.percentile(series, 25)
    q2 = np.percentile(series, 50)
    q3 = np.percentile(series, 75)

    low  = np.clip((q2 - series) / (q2 - q1 + 1e-6), 0, 1)
    med  = np.clip(1 - np.abs(series - q2) / (q3 - q1 + 1e-6), 0, 1)
    high = np.clip((series - q2) / (q3 - q2 + 1e-6), 0, 1)

    return low, med, high

def main():
    df = load_german()

    # fuzzy base vars
    low_s, med_s, high_s = fuzzy_membership_numeric(df["Savings"] if "Savings" in df else df["Saving accounts"].astype("category").cat.codes)
    # better: use credit_amount numeric
    low_c, med_c, high_c = fuzzy_membership_numeric(df["Credit amount"])

    # attach fuzzy membership columns
    df["saving_low"] = low_s
    df["saving_med"] = med_s
    df["saving_high"] = high_s
    df["credit_low"] = low_c
    df["credit_med"] = med_c
    df["credit_high"] = high_c

    # 5 interpretable rules
    df["rule_high_risk_1"] = np.minimum(df["saving_low"], df["credit_high"])
    df["rule_high_risk_2"] = np.minimum(df["saving_low"], df["credit_med"])
    df["rule_low_risk_1"]  = np.minimum(df["saving_high"], df["credit_low"])
    df["rule_low_risk_2"]  = np.minimum(df["saving_med"], df["credit_low"])
    df["rule_low_risk_3"]  = np.minimum(df["saving_high"], df["credit_med"])

    # save for paper evidence
    root = Path(__file__).resolve().parent.parent.parent
    out = root / "Data" / "german_fuzzy_rules.csv"
    df.to_csv(out, index=False)

    print("\nSaved:", out)
    print(df.head(10)[["Risk","saving_low","credit_high","rule_high_risk_1"]])

if __name__ == "__main__":
    main()
