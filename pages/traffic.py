import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os  # Import the os module to check file existence

# Function to load the CSV file containing the labels
def load_labels():
    file_path = "pages/labels.csv"  # Ensure this is the correct path to your CSV file

    # Check if the file exists
    if os.path.exists(file_path):  # Make sure the file path is correct
        try:
            df = pd.read_csv(file_path)  # Load CSV into pandas DataFrame
            # Ensure the CSV contains the correct columns and create a dictionary
            if 'ClassId' in df.columns and 'SignName' in df.columns:
                return dict(zip(df['ClassId'], df['SignName']))  # Return a dictionary of ClassId -> SignName
            else:
                st.error("CSV file must contain 'ClassId' and 'SignName' columns.")
                return None
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.error(f"File '{file_path}' not found.")
        return None

# Function to load the model and make predictions
def predict_image(model, image, class_names):
    # Preprocess the image
    img = image.resize((224, 224))  # Resize to the model's expected input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_probability = predictions[0][predicted_class]

    # Get the predicted class name from the dictionary
    predicted_class_name = class_names.get(predicted_class, "Unknown Sign")

    return predicted_class_name, class_probability, img

# Streamlit UI
st.title("Traffic Sign Classification")
st.write("Upload an image to predict the traffic sign.")

# Load the model and labels (example)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pages/traffic_sign_model.h5")

model = load_model()

# Load class labels from CSV
class_labels = load_labels()

# Check if class labels are successfully loaded
if class_labels is not None:
    st.write("Class Labels Loaded:")
    st.dataframe(pd.DataFrame(list(class_labels.items()), columns=["ClassId", "SignName"]))

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Make prediction
    predicted_class, probability, img = predict_image(model, image, class_labels)

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Probability:** {probability:.2f}")

    # Optionally, display the image using Streamlit
    st.image(img, caption="Processed Image", use_column_width=True)
