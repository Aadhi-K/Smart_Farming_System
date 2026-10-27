import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
CROP_DATA = DATASET_DIR / "Crop_recommendation.csv"
SEED_DATA = DATASET_DIR / "seed_recommendation.csv"
CROP_DETAILS_DATA = DATASET_DIR / "Crop_details.csv"
IRRIGATION_DATA = DATASET_DIR / "irrigation_prediction.csv"
FERTILIZER_DATA = DATASET_DIR / "Fertilizer Prediction.csv"
DISEASE_DATA = DATASET_DIR / "disease.csv"
SEASON_DATA = DATASET_DIR / "season.csv"
CROP_MODEL = MODEL_DIR / "crop_model.pkl"
CROP_ENCODERS = MODEL_DIR / "crop_encoders.pkl"
SEED_MODEL = MODEL_DIR / "seed_model.pkl"
SOIL_MODEL = MODEL_DIR / "soil_model.keras"
HISTORY_FILE = STORAGE_DIR / "history.csv"
SECRET_KEY = os.environ.get("SMARTFARMING_SECRET", "change-this-for-deployment")
SERIAL_PORT = os.environ.get("SMARTFARMING_SERIAL_PORT", "COM3")
BAUD_RATE = 9600
SERIAL_TIMEOUT = 2