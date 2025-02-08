import os
import pandas as pd
import streamlit as st

# Function to load the CSV file containing the labels
def load_labels():
    file_path = "pages/labels.csv"  # Ensure this is the correct path to your CSV file

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)  # Load CSV into pandas DataFrame
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.error(f"File '{file_path}' not found.")
        return None

# Load the labels
class_labels = load_labels()

# If class labels are successfully loaded, display the data
if class_labels is not None:
    st.write("Class Labels Loaded:")
    st.dataframe(class_labels.head())
