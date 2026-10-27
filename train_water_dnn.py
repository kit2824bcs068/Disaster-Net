import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

project_path = Path(__file__).resolve().parent.parent
models_path = project_path / "models"
models_path.mkdir(exist_ok=True)

water_levels = np.random.uniform(0, 100, 1000).reshape(-1, 1)

labels = []

for level in water_levels:
    value = level[0]

    if value < 40:
        labels.append("Normal")
    elif value < 70:
        labels.append("Warning")
    else:
        labels.append("Flood")

labels = np.array(labels)

scaler = StandardScaler()
scaled_levels = scaler.fit_transform(water_levels)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_levels,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

model.save(models_path / "disasternet_water_dnn.keras")

with open(models_path / "water_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open(models_path / "water_label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("\nDNN training completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("Classes:", list(label_encoder.classes_))
print("Model and supporting files saved in the models folder.")