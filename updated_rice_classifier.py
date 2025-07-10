
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load pre-trained model
model = tf.keras.models.load_model("rice_image_classifier.h5")

# Update class names to match your trained model
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']  # Update as per your dataset

st.title("🍚 Rice Grain Classifier")
st.markdown("Upload up to **10 images** of rice grains or use your camera to capture one.")

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "Choose rice image(s)... (Max: 10)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Camera input
camera_image = st.camera_input("Or capture an image using your camera")

# Combine uploaded and camera images into a list
all_images = []

# Enforce upload limit
if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("⚠️ You can upload up to 10 images only. Please remove some files.")
    else:
        all_images.extend(uploaded_files)

if camera_image:
    all_images.append(camera_image)

# Process and predict for each image
for uploaded_file in all_images:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

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
