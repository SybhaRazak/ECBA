import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Function to load class labels from CSV
def load_labels():
    file_path = 'pages/labels.csv'  # Path to your label CSV file
    
    # Load the CSV and check columns
    try:
        df = pd.read_csv(file_path)
        if 'ClassId' in df.columns and 'SignName' in df.columns:
            class_dict = dict(zip(df['ClassId'], df['SignName']))
            return class_dict
        else:
            st.error("CSV file must contain 'ClassId' and 'SignName' columns.")
            return None
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Function to load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("pages/traffic_sign_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict_image(model, image, class_labels):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    predicted_class_name = class_labels.get(predicted_class, "Unknown Sign")
    return predicted_class_name, confidence, image

# Streamlit UI
st.title("Traffic Sign Classification")
st.write("Upload an image to predict the traffic sign.")

# Load class labels
class_labels = load_labels()

# Load the model
model = load_model()

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    if class_labels is not None and model is not None:
        predicted_class, confidence, img = predict_image(model, image, class_labels)
        
        # Display the result
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
