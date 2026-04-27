import os
os.system("pip uninstall -y opencv-python")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# 1. Set up the page layout and title
st.set_page_config(page_title="Resistor Value Detector", layout="centered")
st.title("⚡ Resistor Value Detection App")
st.write("Upload an image of a resistor, and the YOLO specialist model will detect its value and bounding box.")

# 2. Load the model (Cached so it doesn't reload every time a user clicks something)
@st.cache_resource
def load_model():
    # Make sure this filename matches your extracted .pt file exactly
    return YOLO('my_SP_1_Model.pt')

model = load_model()

# 3. Create a file uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns to show original and processed side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        with st.spinner("Detecting resistors..."):
            # Run inference
            results = model.predict(source=image, conf=0.25)
            
            # The .plot() function returns a numpy array in BGR format
            # We use [..., ::-1] to convert it from BGR to RGB for Streamlit to display properly
            result_img = results[0].plot()[..., ::-1]
            
            st.image(result_img, caption="Detection Results", use_container_width=True)
