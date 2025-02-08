import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import os
import time

# Caching the model loading to avoid reloading it multiple times
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("pages/traffic_sign_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Caching the image preprocessing to avoid reprocessing the same image
@st.cache
def preprocess_image(_image):  # Adding _ to bypass caching issues
    image = _image.resize((32, 32))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to load the CSV file containing the labels
def load_labels():
    file_path = "pages/labels.csv"  # Ensure this is the correct path to your CSV file

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)  # Load CSV into pandas DataFrame
            # Convert the DataFrame to a dictionary
            class_dict = dict(zip(df['ClassId'], df['SignName']))
            return class_dict
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    else:
        st.error(f"File '{file_path}' not found.")
        return None

# Function to predict the image
def predict_image(model, image_path, class_names):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = preprocess_image(img)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_probability = predictions[0][predicted_class]

    # Get the predicted class name
    predicted_class_name = class_names.get(predicted_class, "Unknown Sign")

    return predicted_class_name, class_probability, img

# Load model and class labels
model = load_model()
class_labels = load_labels()

if model and class_labels:
    # Upload the image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        try:
            start_time = time.time()
            predicted_class, probability, img = predict_image(model, uploaded_image, class_labels)
            end_time = time.time()

            # Display the prediction results
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Probability: {probability:.2f}")
            st.write(f"Prediction took: {end_time - start_time:.2f} seconds")

            # Display the image
            st.image(img, caption="Processed Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.error("Failed to load model or class labels.")
