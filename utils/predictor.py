import joblib
import pandas as pd
from .config import CROP_MODEL, CROP_ENCODERS, SEED_MODEL, SOIL_MODEL, SEED_DATA


def predict_crop(temperature, humidity, moisture, season, soil):
    try:
        model = joblib.load(CROP_MODEL)
        enc = joblib.load(CROP_ENCODERS)
        season_enc, soil_enc = enc["season"], enc["soil"]
        s = season if season in season_enc.classes_ else season_enc.classes_[0]
        so = soil if soil in soil_enc.classes_ else soil_enc.classes_[0]
        row = pd.DataFrame(
            [
                [
                    temperature,
                    humidity,
                    moisture,
                    season_enc.transform([s])[0],
                    soil_enc.transform([so])[0],
                ]
            ],
            columns=["Temperature", "Humidity", "Moisture", "Season", "Soil"],
        )
        predicted = int(model.predict(row)[0])
        return str(enc["crop"].inverse_transform([predicted])[0])
    except Exception:
        # Transparent fallback for a fresh clone before training.
        return _demo_crop(temperature, moisture, season, soil)


def _demo_crop(temp, moisture, season, soil):
    if soil.lower() in ("clay", "black") and moisture >= 55:
        return "Rice"
    if season.lower() == "winter" and soil.lower() in ("loamy", "clay"):
        return "Wheat"
    if soil.lower() == "sandy":
        return "Groundnut"
    if temp >= 30 and moisture < 50:
        return "Cotton"

    return "Maize"


def predict_seed(crop, season):
    try:
        model = joblib.load(SEED_MODEL)
        return str(model.predict(pd.DataFrame([[crop, season]], columns=["Crop","Season"]))[0])
    except Exception:
        try:
            df = pd.read_csv(SEED_DATA)
            rows = df[
                (df.Crop.str.lower() == crop.lower())
                & (df.Season.str.lower() == season.lower())
            ]
            if rows.empty:
                rows = df[df.Crop.str.lower() == crop.lower()]

            return (
                str(rows.iloc[0]["Seed"])
                if not rows.empty
                else "Consult local seed supplier"
            )
        except Exception:
            return "Consult local seed supplier"


def predict_soil(image_path):
    if not image_path:
        return "Loamy"

    try:
        import tensorflow as tf
        from PIL import Image
        import numpy as np
        model = tf.keras.models.load_model(SOIL_MODEL)
        image = Image.open(image_path).convert("RGB").resize((160, 160))
        arr = np.asarray(image, dtype="float32") / 255.0
        pred = model.predict(arr[None, ...], verbose=0)[0]
        labels = ["Black", "Clay", "Loamy", "Red", "Sandy"]
        return labels[int(np.argmax(pred))]
    except Exception:
        return "Loamy"
