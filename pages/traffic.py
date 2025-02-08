import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import os

# Function to load labels from a CSV file
def load_labels():
    file_path = "pages/labels.csv"  # Update this with the actual path to your CSV file
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            labels_dict = dict(zip(df['ClassId'], df['Name']))  # Assuming 'ClassId' and 'SignName' columns
            return labels_dict
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    else:
        st.error(f"File '{file_path}' not found.")
        return None

# Function to preprocess the image
@st.cache
def preprocess_image(_image):  # Adding _ to bypass caching issues
    image = _image.resize((64, 64))  # Resize image to a size that the model expects, e.g., 64x64
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the traffic sign
def predict_image(model, image_path, class_names):
    # Load and preprocess the image
    img = Image.open(image_path)  # Load the image
    img = preprocess_image(img)  # Preprocess the image
    
    try:
        # Make predictions
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        class_probability = predictions[0][predicted_class]
        predicted_class_name = class_names.get(predicted_class, "Unknown Sign")
        
        return predicted_class_name, class_probability, img

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Assuming the model is in the current directory and named 'traffic_sign_model.h5'
        model = tf.keras.models.load_model("pages/traffic_sign_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main Streamlit interface
st.title("Traffic Sign Recognition")
st.write("Upload an image of a traffic sign and get the prediction.")

# Load model and labels
model = load_model()
class_labels = load_labels()

if model is None or class_labels is None:
    st.stop()

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Show the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Predict the class of the uploaded image
    predicted_class, probability, img = predict_image(model, uploaded_image, class_labels)

    if predicted_class is not None:
        # Display prediction results
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Prediction Probability: {probability:.2f}")
    else:
        st.error("Prediction failed.")
