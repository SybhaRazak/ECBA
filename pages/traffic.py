import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Function to load the CSV file containing the labels
def load_labels():
    file_path = "pages/labels.csv"  # Ensure this is the correct path to your CSV file

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)  # Load CSV into pandas DataFrame
            return dict(zip(df.ClassId, df.SignName))  # Return as a dictionary
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.error(f"File '{file_path}' not found.")
        return None

# Load the labels
class_labels = load_labels()

# If class labels are successfully loaded, display the data
if class_labels is not None:
    st.write("Class Labels Loaded:")
    st.write(class_labels)  # Display the loaded class labels as a dictionary

# Load trained model
def load_model():
    model_path = "pages/traffic_sign_model.h5"
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error(f"Model file '{model_path}' not found.")
        return None

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    try:
        # Check if the image is in RGB format, else convert
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match model input size (32x32 for example)
        image = image.resize((32, 32))
        
        # Convert to numpy array and normalize
        image = np.array(image) / 255.0
        
        # Check if the image has 3 channels (RGB)
        if image.shape[-1] != 3:
            raise ValueError("Image must have 3 channels (RGB)")

        # Add batch dimension (1, 32, 32, 3)
        image = np.expand_dims(image, axis=0)
        return image

    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

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
    if processed_image is not None:
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with the highest confidence
        confidence = np.max(prediction)  # Get the confidence of the prediction
        
        # Display results
        sign_name = class_labels.get(predicted_class, "Unknown Sign")
        st.write(f"**Prediction:** {sign_name}")
        st.write(f"**Confidence:** {confidence:.2f}")



