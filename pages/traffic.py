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
st.write("Loaded Class Labels:", class_labels)  # Debugging output

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
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]  # Get top 3 predictions
        top_3_confidences = prediction[0][top_3_indices] * 100
        
        # Debugging Output for Predictions
        st.write(f"Raw Prediction Output: {prediction}")
        st.write(f"Top-3 Predictions:")
        for i in range(3):
            class_id = top_3_indices[i]
            class_name = class_labels.get(class_id, 'Unknown')
            confidence = top_3_confidences[i]
            st.write(f"{i+1}. {class_name} ({confidence:.2f}%)")
        
        # Display top prediction
        st.write(f"**Final Prediction:** {class_labels.get(top_3_indices[0], 'Unknown')}")
        st.write(f"**Confidence:** {top_3_confidences[0]:.2f}%")
    except ValueError as e:
        st.error(f"Model Prediction Error: {e}")
