import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Load class labels from label.csv
@st.cache_data
def load_labels():
    df = pd.read_csv("label.csv")  # Ensure the CSV has 'ClassId' and 'SignName' columns
    return dict(zip(df.ClassId, df.SignName))

class_labels = load_labels()

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_model.h5")

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Traffic Sign Classification")
st.write("Upload an image to predict the traffic sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    # Display results
    sign_name = class_labels.get(predicted_class, "Unknown Sign")
    st.write(f"**Prediction:** {sign_name}")
    st.write(f"**Confidence:** {confidence:.2f}")
