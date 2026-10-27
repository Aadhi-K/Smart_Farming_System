import pandas as pd
from .config import CROP_DETAILS_DATA, IRRIGATION_DATA, FERTILIZER_DATA, DISEASE_DATA, SEASON_DATA


def _read(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def season_for_month(month):
    df = _read(SEASON_DATA)
    if not df.empty:
        match = df[df["Month"].astype(str).str.lower() == str(month).lower()]
        if not match.empty:
            return str(match.iloc[0]["Season"])

    return "Monsoon"


def crop_details(crop):
    df = _read(CROP_DETAILS_DATA)
    if df.empty:
        return {"Description": "No crop details available."}

    match = df[df["Crop"].astype(str).str.lower() == crop.lower()]
    if not match.empty:
        return match.iloc[0].to_dict()

    return {"Description": "Details not found."}


def irrigation_advice(moisture):
    df = _read(IRRIGATION_DATA)
    if df.empty:
        return {
            "Pump_Status": "UNKNOWN",
            "Recommendation": "Irrigation data unavailable.",
        }

    row = df[(df["Moisture_Min"] <= moisture) & (df["Moisture_Max"] >= moisture)]
    if not row.empty:
        return row.iloc[0].to_dict()

    return {
        "Pump_Status": "UNKNOWN",
        "Recommendation": "Check moisture sensor.",
    }


def fertilizer_advice(crop):
    df = _read(FERTILIZER_DATA)
    if df.empty:
        return []

    rows = df[df["Crop"].astype(str).str.lower() == crop.lower()]
    return rows.to_dict("records")


def disease_info(crop):
    df = _read(DISEASE_DATA)
    if df.empty:
        return []

    rows = df[df["Crop"].astype(str).str.lower() == crop.lower()]
    return rows.to_dict("records")