import streamlit as st
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Define the layout of the app
st.set_page_config(page_title="testmodel1", page_icon=":medical_symbol:")

st.title("testmodel1")

st.header("Dementia Clock Test 90% accurate")

st.write("Try drawing a clock")

st.caption("Detections available: Healthy, Dementia and empty")

st.warning("your drawing wouldn't be saved, so if you want to draw you can do it on another website/app")

with st.sidebar:
    img = Image.open("./Images/parkinson_disease_detection.jpg")
    st.image(img)
    st.subheader("About testmodel1")
    link_text = "Distinguishing Different Stages of Parkinson’s Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral"
    link_url = "https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full"
    st.write(
        "Parkinson's disease is a neurodegenerative disorder that affects motor functions, leading to tremors, stiffness, and impaired movement. The research presented in the article link mentioned below explores the use of spiral and wave sketch images to develop a robust algorithm for Parkinson's disease detection. testmodel1 leverages these sketch images to train an AI model, achieving an impressive accuracy rate of 83%."
    )
    st.markdown(f"[{link_text}]({link_url})")
    st.header("Dataset")
    img = Image.open("./Images/healthy_diseased_classification.jpeg")
    st.image(img)
    st.header("Drawing Canvas Configurations")

# Specify canvas parameters in application
drawing_mode = "freedraw"

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Split the layout into two columns
col1, col2 = st.columns(2)

# Define the canvas size
canvas_size = 500  # Update the canvas size to 500 pixels

with col1:
    # Create a canvas component
    st.subheader("Drawable Canvas")
    canvas_image = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=canvas_size,
        height=canvas_size,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )

with col2:
    st.subheader("Preview")
    if canvas_image.image_data is not None:
        # Get the numpy array (4-channel RGBA 100,100,4)
        input_numpy_array = np.array(canvas_image.image_data)
        # Get the RGBA PIL image
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        st.image(input_image, use_column_width=True)

# Rest of your code...
