
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load pre-trained model
model = tf.keras.models.load_model("rice_classifier.h5")

# Update class names to match your trained model
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']  # Update as per your dataset

st.title("🍚 Rice Grain Classifier")
st.markdown("Upload an image of a rice grain and the model will predict its type.")

uploaded_file = st.file_uploader("Choose a rice image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.success(f"Predicted Rice Type: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
