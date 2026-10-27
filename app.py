import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from google import genai

st.set_page_config(
    page_title="DisasterNet",
    page_icon="🌊",
    layout="centered"
)

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_path, "models")

model = tf.keras.models.load_model(
    os.path.join(models_path, "disasternet_water_dnn.keras")
)

with open(os.path.join(models_path, "water_scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

with open(os.path.join(models_path, "water_label_encoder.pkl"), "rb") as file:
    label_encoder = pickle.load(file)

st.title("🌊 DisasterNet")
st.subheader("Multimodal Flood / Landslide Early Warning")

st.divider()

water_level = st.slider(
    "Water Level (%)",
    min_value=0,
    max_value=100,
    value=20
)

tilt_status = st.selectbox(
    "Tilt Sensor Status",
    ["Normal", "Abnormal Movement"]
)

if st.button("🔍 Analyze Situation", use_container_width=True):

    data = np.array([[water_level]])
    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data, verbose=0)

    predicted_class = np.argmax(prediction)
    dnn_result = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = prediction[0][predicted_class] * 100

    if tilt_status == "Abnormal Movement":
        hazard = "Landslide Movement Warning"
    elif dnn_result == "Flood":
        hazard = "Flood"
    elif dnn_result == "Warning":
        hazard = "Flood Warning"
    else:
        hazard = "Normal"

    st.divider()

    st.metric("Water Level", f"{water_level}%")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### DNN Detection")
        st.write(dnn_result)
        st.write(f"Confidence: {confidence:.2f}%")

    with col2:
        st.write("### Tilt Status")
        st.write(tilt_status)

    st.write("### 🚨 Hazard Status")
    st.warning(hazard)

    st.divider()

    st.write("### 🤖 GenAI Safety Response")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY is not set.")
    else:

        client = genai.Client(api_key=api_key)

        prompt = f"""
You are the response generator for DisasterNet,
a low-cost IoT disaster early-warning prototype.

Sensor information:
Water level: {water_level}%
DNN detection: {dnn_result}
Tilt status: {tilt_status}
Final hazard: {hazard}

Give a short response containing:
1. Hazard status
2. One brief explanation
3. Three practical safety actions

Do not invent sensor readings.
Do not claim this prototype can accurately predict real disasters.
Keep the response suitable for a college project dashboard.
"""

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            st.info(response.text)

        except Exception as error:
            st.error(f"GenAI error: {error}")

st.divider()

st.caption(
    "DisasterNet is a college-level prototype for demonstrating "
    "IoT sensing, DNN classification and GenAI response generation."
)