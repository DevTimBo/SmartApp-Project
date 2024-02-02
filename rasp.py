import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os
import platform
from rmback import isolate_paper




# Function to capture and save image
def capture_and_save_image(camera, file_path):
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(frame)
    captured_image.save(file_path)
    return file_path


def start_run_pipeline(file_path):
    # Check if the file already exists and show a link to view it
    if os.path.exists(file_path):
        #run_pipeline(file_path)
         return file_path

# Set a default file path for saving images
file_path = "captured_image.png"


# Create a folder for captured images if it doesn't exist
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Create a folder-specific file path
file_path = os.path.join("captured_images", "captured_image.png")

isRun =False

run_checkbox_key = "run_camera"
st.title("Smart App")
run = st.checkbox('Kamera starten',key=run_checkbox_key)
capture_button = st.button('Bild aufnehmen', key='capture')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Create the checkbox outside the loop

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
    if capture_button:
        # Capture and save the image
        saved_file_path = capture_and_save_image(camera, file_path)
        st.success(f"Bild erfolgreich aufgenommen und gespeichert: [Bild anzeigen]({saved_file_path})")
        capture_button = False  

# This line will only be reached when run is False

st.write('Kamera aus')
camera.release()
# Check if the file already exists and show a link to view it
if os.path.exists(file_path):
    st.success(f"Vorheriges Bild vorhanden: [Bild anzeigen]({file_path})")
    captured_image = Image.open(file_path)
    st.image(captured_image, caption='Captured Image')
    isolate_paper(file_path, file_path)
    pipeline_start_button = st.button('Bearbeitung starten', key='pipeline', on_click=start_run_pipeline(file_path))


  
       