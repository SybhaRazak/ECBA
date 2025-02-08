import os
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Function to load class labels from a CSV file
def load_labels():
    class_labels = {}
    try:
        file_path = "pages/labels.csv"  # Replace with your actual file path
        if os.path.exists(file_path):
            import pandas as pd
            df = pd.read_csv(file_path)
            class_labels = pd.Series(df.Name.values, index=df.ClassId).to_dict()
        else:
            st.error(f"Class labels CSV file '{file_path}' not found.")
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
    return class_labels

# Function to load the model with caching
@st.cache_resource
def load_model():
    try:
        model_path = 'pages/traffic_sign_model.h5'  # Replace with your actual model path
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.error(f"Model file not found at {model_path}.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the image (resize, scale, and expand dimensions for prediction)
def preprocess_image(image):
    try:
        # Resize image to 64x64 (adjust according to the model input size)
        image = image.resize((64, 64))  # Resize to match the model input size (64x64 is assumed)
        img_array = np.array(image)
        
        # If the image has only 1 channel (grayscale), we need to convert it to 3 channels
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

def preprocess_image(image):
    try:
        # Resize image to a larger size (e.g., 128x128)
        image = image.resize((128, 128))  # Resize to match the model input size (128x128 is assumed)
        img_array = np.array(image)
        
        # If the image has only 1 channel (grayscale), we need to convert it to 3 channels
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Main Streamlit code
def main():
    # Load class labels
    class_labels = load_labels()
    if not class_labels:
        st.error("Class labels could not be loaded. Exiting.")
        return

    # Load the model
    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Exiting.")
        return

    # Upload image
    st.title("Traffic Sign Prediction")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Predict and display the result
        predicted_class, probability, img = predict_image(model, image, class_labels)
        if predicted_class:
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Probability: {probability * 100:.2f}%")
            
            # Display the image with the predicted label
            st.image(img, caption=f"Prediction: {predicted_class}", use_column_width=True)

# Run the main function
if __name__ == "__main__":
    main()
