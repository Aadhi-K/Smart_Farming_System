"""Optional TensorFlow image classifier. Expected folders: datasets/soil_images/{Black,Clay,Loamy,Red,Sandy}/"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_DIR, SOIL_MODEL

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit("Install TensorFlow first: pip install tensorflow") from e

image_dir = DATASET_DIR / "soil_images"
if not image_dir.exists():
    raise SystemExit(f"Missing folder: {image_dir}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(160, 160),
    batch_size=16,
    label_mode="categorical",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(160, 160),
    batch_size=16,
    label_mode="categorical",
)
class_names = train_ds.class_names
if len(class_names) < 2:
    raise SystemExit("Create at least two class folders with labelled images.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
).prefetch(AUTOTUNE)
val_ds = val_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
).prefetch(AUTOTUNE)
base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False, weights="imagenet"
)
base.trainable = False
inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=8)
SOIL_MODEL.parent.mkdir(parents=True, exist_ok=True)
model.save(SOIL_MODEL)
print("Saved:", SOIL_MODEL)
print("Soil class order:", class_names)
print("IMPORTANT: predictor.py uses a fixed class order; keep folder names/order aligned: Black, Clay, Loamy, Red, Sandy.")
