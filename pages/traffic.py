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
            class_labels = pd.Series(df.SignName.values, index=df.ClassId).to_dict()
        else:
            st.error(f"Class labels CSV file '{file_path}' not found.")
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
    return class_labels

# Function to load the model
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
        image = image.resize((32, 32))  # Resize to match the model input
        img_array = np.array(image)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Function to make predictions
def predict_image(model, image, class_names):
    try:
        processed_image = preprocess_image(image)
        if processed_image is not None:
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            predicted_class_name = class_names.get(predicted_class, "Unknown Sign")
            class_probability = predictions[0][predicted_class]
            return predicted_class_name, class_probability, image
        else:
            st.error("Image preprocessing failed.")
            return None, None, None
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

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
