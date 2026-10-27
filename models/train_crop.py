"""Train crop recommendation model from datasets/Crop_recommendation.csv."""
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CROP_DATA, CROP_MODEL, CROP_ENCODERS

required = ["Temperature", "Humidity", "Moisture", "Season", "Soil", "Crop"]
df = pd.read_csv(CROP_DATA)
missing = set(required) - set(df.columns)
if missing:
	raise ValueError(f"CSV missing columns: {sorted(missing)}")

df = df.dropna(subset=required).copy()
if len(df) < 10:
	raise ValueError("Add more labelled rows before training (at least 10 for this demo).")

season_enc, soil_enc, crop_enc = LabelEncoder(), LabelEncoder(), LabelEncoder()
df["Season"] = season_enc.fit_transform(df["Season"].astype(str))
df["Soil"] = soil_enc.fit_transform(df["Soil"].astype(str))
y = crop_enc.fit_transform(df["Crop"].astype(str))
X = df[["Temperature", "Humidity", "Moisture", "Season", "Soil"]]
stratify = y if pd.Series(y).value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.25, random_state=42, stratify=stratify
)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Demo holdout accuracy:", round(accuracy_score(y_test, pred), 3))
print(
	classification_report(
		y_test,
		pred,
		labels=range(len(crop_enc.classes_)),
		target_names=crop_enc.classes_,
		zero_division=0,
	)
)
CROP_MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, CROP_MODEL)
joblib.dump({"season": season_enc, "soil": soil_enc, "crop": crop_enc}, CROP_ENCODERS)
print("Saved:", CROP_MODEL, "and", CROP_ENCODERS)
print("WARNING: example CSV is synthetic/tiny; metrics are not evidence of real-world accuracy.")
