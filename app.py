from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
from utils.config import SECRET_KEY, UPLOAD_FOLDER, HISTORY_FILE
from utils.predictor import predict_crop, predict_seed, predict_soil
from utils.csv_loader import (
    crop_details,
    disease_info,
    fertilizer_advice,
    irrigation_advice,
    season_for_month,
)
from utils.sensor import latest_sensor

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Path(HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", sensor=latest_sensor())


@app.route("/sensor")
def sensor_api():
    return latest_sensor()


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", month="June", sensor=latest_sensor())
    try:
        temperature = float(request.form.get("temperature", 29))
        humidity = float(request.form.get("humidity", 70))
        moisture = float(request.form.get("moisture", 55))
        soil = request.form.get("soil", "Loamy").strip().title()
        soil_mode = request.form.get("soil_mode", "known")
        month = request.form.get("month", "June")
        season = season_for_month(month)

        if soil_mode not in ("known", "photo"):
            raise ValueError("Choose whether to enter a soil type or provide a soil photo.")

        image = request.files.get("soil_image") if soil_mode == "photo" else None
        if soil_mode == "photo" and (not image or not image.filename):
            raise ValueError("Choose or capture a soil photo to identify the soil type.")

        soil_image_path = None
        if image and image.filename:
            safe = secure_filename(image.filename)
            soil_image_path = UPLOAD_FOLDER / safe
            image.save(soil_image_path)
        detected_soil = predict_soil(soil_image_path) if soil_image_path else soil
        crop = predict_crop(temperature, humidity, moisture, season, detected_soil)
        seed = predict_seed(crop, season)
        details = crop_details(crop)
        irrigation = irrigation_advice(moisture)
        fertilizer = fertilizer_advice(crop)
        disease = disease_info(crop)
        result = {
            "crop": crop,
            "seed": seed,
            "soil": detected_soil,
            "temperature": temperature,
            "humidity": humidity,
            "moisture": moisture,
            "season": season,
            "details": details,
            "irrigation": irrigation,
            "fertilizer": fertilizer,
            "disease": disease,
        }
        row = {
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Crop": crop,
            "Seed": seed,
            "Soil": detected_soil,
            "Season": season,
            "Temperature": temperature,
            "Humidity": humidity,
            "Moisture": moisture,
        }
        history = Path(HISTORY_FILE)
        pd.DataFrame([row]).to_csv(history, mode="a", header=not history.exists(), index=False)
        return render_template("result.html", result=result)
    except Exception as exc:
        flash(f"Prediction could not be completed: {exc}", "error")
        return redirect(url_for("predict"))


@app.route("/history")
def history():
    path = Path(HISTORY_FILE)
    records = pd.read_csv(path).to_dict("records") if path.exists() and path.stat().st_size else []
    records.reverse()
    return render_template("history.html", history=records)


if __name__ == "__main__":
    app.run(debug=True)