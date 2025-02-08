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

# Streamlit Webpage Layout
st.title("Traffic Sign Classification")
st.write("Upload an image to classify the traffic sign.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Prediction process when an image is uploaded
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
        # Make prediction
        prediction = model.predict(img_array)
        
        # Debugging: Show raw prediction
        st.write(f"Raw Prediction Output: {prediction}")
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]  # Get top 3 predictions
        top_3_confidences = prediction[0][top_3_indices] * 100
        
        # Display predictions with confidence
        st.write(f"Top-3 Predictions:")
        for i in range(3):
            class_id = top_3_indices[i]
            class_name = class_labels.get(class_id, 'Unknown')
            confidence = top_3_confidences[i]
            st.write(f"{i+1}. **{class_name}** ({confidence:.2f}%)")
        
        # Display the final prediction
        final_prediction = class_labels.get(top_3_indices[0], 'Unknown')
        final_confidence = top_3_confidences[0]
        
        st.write(f"### Final Prediction:")
        st.markdown(f"**{final_prediction}** with confidence of **{final_confidence:.2f}%**")
        
        # Optionally add a check if the top prediction is above a certain confidence threshold
        if final_confidence < 60:  # Adjust threshold if needed
            st.warning(f"The model is not confident in this prediction. Please try another image.")
        else:
            st.success(f"Prediction is **{final_prediction}** with high confidence!")
    
    except ValueError as e:
        st.error(f"Error occurred during prediction: {e}")

else:
    st.info("Please upload an image to start the prediction process.")
