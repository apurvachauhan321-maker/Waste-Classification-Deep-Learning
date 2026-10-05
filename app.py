import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/waste_classifier.keras")

# Classes
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Page settings
st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️"
)

st.title("♻️ Waste Classification Using Deep Learning")
st.write("Upload an image of waste and the CNN model will predict its category.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # Prepare image
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class.capitalize()}")
    st.info(f"Confidence: {confidence:.2f}%")