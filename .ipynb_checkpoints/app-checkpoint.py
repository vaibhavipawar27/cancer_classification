import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained CNN model
model = load_model("cancer_cnn_model.h5")

# Define class labels
CLASS_NAMES = ["Benign", "Malignant"]

st.title("🧬 Cancer Scan Classifier")
st.write("Upload a histopathology or cancer scan image to predict.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Scan Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Adjust as per model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # Show result
    st.subheader("🎯 Prediction:")
    st.write(f"**{CLASS_NAMES[class_index]}** with {confidence:.2f}% confidence.")
