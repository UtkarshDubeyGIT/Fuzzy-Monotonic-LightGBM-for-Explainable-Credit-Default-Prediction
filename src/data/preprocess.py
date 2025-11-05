"""
COPILOT TASK — UPDATE PREPROCESS PIPELINE

Goal:
Enhance feature engineering for Taiwan dataset to support fuzzy + monotonic modeling.

Steps to modify inside this file:

1) After reading and cleaning dataset:
   create engineered features (train_only_fit if needed):
      - BILL_AMT_AVG = mean of BILL_AMT1..BILL_AMT6
      - utilization = BILL_AMT_AVG / LIMIT_BAL  (clip 0→1)
      - repay_ratio1 = PAY_AMT1 / BILL_AMT1     (handle zero division → fill 0, clip 0→1)
      - delinquency_intensity = max(PAY_0..PAY_6)
      - paytrend = (PAY_AMT6 – PAY_AMT1) / (PAY_AMT1 + 1e-6) (clip to -1→1)

2) these engineered cols MUST be included in the final feature set output to model training.

3) continue applying existing scaling + train/test split as before.

4) do NOT drop original columns yet. We keep originals + engineered features.

5) return/save processed train/test CSVs with these new columns added.

Expected final new columns:
    BILL_AMT_AVG
    utilization
    repay_ratio1
    delinquency_intensity
    paytrend

DO NOT add model training code here. ONLY feature engineering + preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


def _read_taiwan_csv(data_path: Path) -> pd.DataFrame:
	"""Read the Taiwan credit dataset, handling the extra header row and unnamed index.

	The file at src/data/taiwan_credit.csv contains an initial mapping/header row
	like [ , X1..X23, Y] followed by the proper header row
	[ID, LIMIT_BAL, SEX, ..., default payment next month]. We therefore set header=1
	and drop any unnamed columns if present.
	"""
	df = pd.read_csv(data_path, header=1)
	# Drop any unnamed index columns if present
	df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]  
	return df


def _fit_label_encoders(X_train: pd.DataFrame) -> Tuple[Dict[str, Dict[str, int]], list[str]]:
	"""Fit per-column label encoders on object-dtype columns using train data only.

	Returns (encoders, cat_cols) where encoders is mapping of
	column -> {label_str: encoded_int}. Unknown labels at transform time map to -1.
	"""
	encoders: Dict[str, Dict[str, int]] = {}
	cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']
	for col in cat_cols:
		ser = X_train[col].fillna('__MISSING__').astype(str)
		le = LabelEncoder()
		le.fit(ser)
		classes_map = {cls: idx for idx, cls in enumerate(le.classes_)}
		encoders[col] = classes_map
	return encoders, cat_cols


def _apply_label_encoders(df: pd.DataFrame, encoders: Dict[str, Dict[str, int]]) -> pd.DataFrame:
	"""Apply pre-fitted label encoder mappings to a dataframe; unseen -> -1."""
	for col, mapping in encoders.items():
		df[col] = (
			df[col]
			.fillna('__MISSING__')
			.astype(str)
			.map(mapping)
			.fillna(-1)
			.astype(int)
		)
	return df


def _fit_scaler(X_train: pd.DataFrame, num_cols: list[str]) -> RobustScaler:
	"""Fit RobustScaler on provided numeric columns of X_train only and return scaler."""
	scaler = RobustScaler()
	if num_cols:
		scaler.fit(X_train[num_cols])
	return scaler


def _apply_scaler(X: pd.DataFrame, scaler: RobustScaler, num_cols: list[str]) -> pd.DataFrame:
	if not num_cols:
		return X
	X_scaled = X.copy()
	X_scaled[num_cols] = scaler.transform(X[num_cols])
	return X_scaled


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Add engineered features required for fuzzy + monotonic modeling.

	Features:
	- BILL_AMT_AVG: mean of BILL_AMT1..BILL_AMT6
	- utilization: BILL_AMT_AVG / LIMIT_BAL (clipped 0→1; division by zero -> 0)
	- repay_ratio1: PAY_AMT1 / BILL_AMT1 (clipped 0→1; division by zero -> 0)
	- delinquency_intensity: row-wise max over PAY_0..PAY_6 (only existing cols)
	- paytrend: (PAY_AMT6 – PAY_AMT1) / (PAY_AMT1 + 1e-6) (clipped -1→1)
	"""
	out = df.copy()

	# BILL_AMT_AVG
	bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in out.columns]
	if bill_cols:
		out['BILL_AMT_AVG'] = out[bill_cols].mean(axis=1)
	else:
		out['BILL_AMT_AVG'] = 0.0

	# utilization (clip 0..1)
	if 'LIMIT_BAL' in out.columns:
		util = (out['BILL_AMT_AVG'] / out['LIMIT_BAL'].replace(0, pd.NA)).fillna(0.0)
		out['utilization'] = util.clip(lower=0, upper=1)
	else:
		out['utilization'] = 0.0

	# repay_ratio1 (clip 0..1)
	if 'PAY_AMT1' in out.columns and 'BILL_AMT1' in out.columns:
		ratio = (out['PAY_AMT1'] / out['BILL_AMT1'].replace(0, pd.NA)).fillna(0.0)
		out['repay_ratio1'] = ratio.clip(lower=0, upper=1)
	else:
		out['repay_ratio1'] = 0.0

	# delinquency_intensity = max PAY_0..PAY_6 over available columns
	pay_cols = [f'PAY_{i}' for i in range(0, 7) if f'PAY_{i}' in out.columns]
	if pay_cols:
		out['delinquency_intensity'] = out[pay_cols].max(axis=1)
	else:
		out['delinquency_intensity'] = 0

	# paytrend (clip -1..1)
	if 'PAY_AMT6' in out.columns and 'PAY_AMT1' in out.columns:
		trend = (out['PAY_AMT6'] - out['PAY_AMT1']) / (out['PAY_AMT1'] + 1e-6)
		out['paytrend'] = trend.clip(lower=-1, upper=1).fillna(0.0)
	else:
		out['paytrend'] = 0.0

	return out

def load_and_preprocess_taiwan(
	data_path: str | Path | None = None,
	*,
	test_size: float = 0.2,
	random_state: int = 42,
	save: bool = True,
	engineer_features: bool = True,
	output_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Load, clean, and split the Taiwan default dataset with leakage safety.

	Steps (strict rules):
	1) Rename target column 'default payment next month' -> 'default'
	2) Convert target to binary integers (0/1)
	3) Encode categorical (object-dtype) columns using LabelEncoder (fit ONLY on train)
	4) Apply RobustScaler on numeric columns (fit ONLY on train)
	5) Create 80/20 stratified train-test split
	6) Save processed train/test to data/processed/{train,test}.csv (if save=True)
	7) Return X_train, X_test, y_train, y_test objects
	8) Only scale numeric columns (no blind float casting)
	9) No metrics are computed here

	No resampling occurs here (class balancing is for later modeling stage).
	"""

	# Resolve default path relative to this file to keep importable/reusable behavior
	here = Path(__file__).resolve()
	if data_path is None:
		data_path = here.parent / 'taiwan_credit.csv'
	else:
		data_path = Path(data_path)

	# Load raw
	df = _read_taiwan_csv(data_path)

	# Standardize target name
	if 'default payment next month' in df.columns:
		df = df.rename(columns={'default payment next month': 'default'})
	elif 'Y' in df.columns and 'default' not in df.columns:
		df = df.rename(columns={'Y': 'default'})

	if 'default' not in df.columns:
		raise ValueError("Target column 'default' not found after renaming.")

	# Convert target to binary integers
	df['default'] = df['default'].astype(int)

	# Optional leakage safety: drop pure identifiers if present
	if 'ID' in df.columns:
		df = df.drop(columns=['ID'])

	# Split early to fit encoders/scalers ONLY on train
	X = df.drop(columns=['default'])
	y = df['default']

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)

	# Add engineered features BEFORE encoding + scaling (optional)
	if engineer_features:
		X_train = _add_engineered_features(X_train)
		X_test = _add_engineered_features(X_test)

	# Encode categorical object-dtype columns on train only
	encoders, cat_cols = _fit_label_encoders(X_train)
	X_train = _apply_label_encoders(X_train, encoders)
	X_test = _apply_label_encoders(X_test, encoders)

	# Scale numeric columns (fit on train, apply to both), excluding encoded categoricals
	num_cols = [c for c in X_train.columns if is_numeric_dtype(X_train[c]) and c not in cat_cols]
	scaler = _fit_scaler(X_train, num_cols)
	if num_cols:
		X_train = _apply_scaler(X_train, scaler, num_cols)
		X_test = _apply_scaler(X_test, scaler, num_cols)

	# Persist processed splits if requested
	if save:
		if output_dir is None:
			repo_root = here.parent.parent.parent  # repo/
			out_dir = repo_root / 'data' / 'processed'
		else:
			out_dir = Path(output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

		train_df = X_train.copy()
		train_df['default'] = y_train.values

		test_df = X_test.copy()
		test_df['default'] = y_test.values

		train_df.to_csv(out_dir / 'train.csv', index=False)
		test_df.to_csv(out_dir / 'test.csv', index=False)

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":
	import argparse
	import warnings
	from pathlib import Path

	# CLI: do work silently and only raise errors if any
	parser = argparse.ArgumentParser(description="Preprocess Taiwan dataset")
	parser.add_argument("--engineer-features", choices=["0", "1"], required=True,
						help="1 to add engineered features, 0 to skip")
	parser.add_argument("--output-dir", type=str, required=True,
						help="Directory where train.csv and test.csv will be saved")
	parser.add_argument("--variant", type=str, required=False, help="Variant name (unused)")
	args = parser.parse_args()

	engineer_features = bool(int(args.engineer_features))
	output_dir = Path(args.output_dir).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	# Silence non-error warnings to avoid any stdout/stderr noise
	warnings.filterwarnings("ignore")

	# Run preprocessing and always save into output_dir. No prints.
	load_and_preprocess_taiwan(
		data_path=None,
		test_size=0.2,
		random_state=42,
		save=True,
		engineer_features=engineer_features,
		output_dir=output_dir,
	)

