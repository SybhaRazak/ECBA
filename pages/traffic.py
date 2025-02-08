import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pages/traffic_sign_model.h5")  # Update with actual model path
    return model

model = load_model()

# Define class labels (update based on dataset used)
class_labels = {0: "Stop", 1: "Yield", 2: "Speed Limit", 3: "No Entry"}  # Example labels

st.title("Traffic Sign Classification")
st.write("Upload an image to classify the traffic sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.resize((32, 32))  # Update with actual model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    st.write(f"**Prediction:** {class_labels.get(predicted_class, 'Unknown')}")
    st.write(f"**Confidence:** {confidence:.2f}%")
