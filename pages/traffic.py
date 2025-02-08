import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# Function to load the CSV file containing the labels
def load_labels():
    file_path = "pages/labels.csv"  # Ensure this is the correct path to your CSV file

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)  # Load CSV into pandas DataFrame
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.error(f"File '{file_path}' not found.")
        return None

# If class labels are successfully loaded, display the data
if class_labels is not None:
    st.write("Class Labels Loaded:")
    st.dataframe(class_labels.head()) 

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pages/traffic_sign_model.h5")

model = load_model()

# Function to preprocess the image with a larger size (64x64)
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to a larger size (64x64)
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 64, 64, 3)
    st.write(f"Processed Image Shape: {image.shape}")  # Debugging the shape of the image
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
    try:
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        # Display results
        sign_name = class_labels.get(predicted_class, "Unknown Sign")
        st.write(f"**Prediction:** {sign_name}")
        st.write(f"**Confidence:** {confidence:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

