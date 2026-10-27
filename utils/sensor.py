from .config import SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT


def read_sensor_once():
    try:
        import serial

        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT) as ser:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        values = [float(x.strip()) for x in line.split(",")]
        if len(values) != 3:
            raise ValueError("Expected temperature,moisture,humidity")

        temperature, moisture, humidity = values
        return {
            "temperature": temperature,
            "moisture": moisture,
            "humidity": humidity,
            "source": "ESP32",
        }
    except Exception:
        return None


def latest_sensor():
    data = read_sensor_once()
    if data:
        return data

    return {
        "temperature": 34,
        "humidity": 80,
            "moisture": 60,
        "source": "DEMO (serial unavailable)",
    }