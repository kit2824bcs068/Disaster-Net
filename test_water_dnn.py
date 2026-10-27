import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent
models_path = project_path / "models"

model = tf.keras.models.load_model(
    models_path / "disasternet_water_dnn.keras"
)

with open(models_path / "water_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open(models_path / "water_label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

test_levels = [20, 55, 85]

for water_level in test_levels:

    data = np.array([[water_level]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data, verbose=0)

    predicted_class = np.argmax(prediction)

    label = label_encoder.inverse_transform([predicted_class])[0]

    confidence = prediction[0][predicted_class] * 100

    print(
        f"Water Level: {water_level}% | "
        f"Prediction: {label} | "
        f"Confidence: {confidence:.2f}%"
    )