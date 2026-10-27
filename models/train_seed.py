"""Seed recommendation is a small categorical lookup problem; this script trains a simple classifier."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SEED_DATA, SEED_MODEL

df = pd.read_csv(SEED_DATA).dropna(subset=["Crop", "Season", "Seed"])
if len(df) < 8:
	raise ValueError("Add more rows to seed_recommendation.csv before training.")

X, y = df[["Crop", "Season"]].astype(str), df["Seed"].astype(str)
prep = ColumnTransformer(
	[
		(
			"categorical",
			OneHotEncoder(handle_unknown="ignore"),
			["Crop", "Season"],
		)
	]
)
model = Pipeline(
	[
		("prep", prep),
		(
			"classifier",
			RandomForestClassifier(n_estimators=150, random_state=42),
		),
	]
)
# A holdout score on tiny one-row-per-combination data is not reliable; train on all rows for lookup coverage.
model.fit(X, y)
SEED_MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, SEED_MODEL)
print(f"Saved seed model: {SEED_MODEL} (trained on {len(df)} rows).")
print("Note: for exact crop-season mappings, CSV lookup is often more appropriate than ML.")
