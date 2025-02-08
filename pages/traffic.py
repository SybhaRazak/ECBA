import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pages/traffic_sign_model.h5")  # Update with actual model path
    return model

model = load_model()

# Load class labels from CSV
@st.cache_resource
def load_labels():
    labels_df = pd.read_csv("pages/labels.csv")  # Ensure this path is correct
    return {row["ClassId"]: row["Name"] for _, row in labels_df.iterrows()}  # Adjust column names if needed

class_labels = load_labels()

st.title("Traffic Sign Classification")
st.write("Upload an image to classify the traffic sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get model input shape dynamically
    input_shape = model.input_shape[1:3]  # Extract expected height and width
    st.write(f"Expected Model Input Shape: {input_shape}")
    
    # Preprocess image
    img = image.resize(input_shape)  # Resize to model's expected input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Debugging Output
    st.write(f"Image Shape Before Model: {img_array.shape}")
    
    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.write(f"**Prediction:** {class_labels.get(predicted_class, 'Unknown')}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    except ValueError as e:
        st.error(f"Model Prediction Error: {e}")
