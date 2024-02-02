import platform
import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np

# Check platform to determine the camera module
is_raspberry_pi = platform.machine().startswith("arm")  # Assumes Raspberry Pi is ARM-based

if not is_raspberry_pi:
    st.error("This code is intended to run on Raspberry Pi only.")
    st.stop()

from picamera import PiCamera
from picamera.array import PiRGBArray

# Function to capture and save image
def capture_and_save_image(camera, file_path):
    with PiRGBArray(camera) as stream:
        camera.capture(stream, format='rgb')
        frame = stream.array

    if frame is not None:
        captured_image = Image.fromarray(frame)
        captured_image.save(file_path)
        return file_path
    else:
        st.error("Failed to capture image.")
        return None

def start_run_pipeline(file_path):
    # Check if the file already exists and show a link to view it
    if os.path.exists(file_path):
        # run_pipeline(file_path)
        return file_path

# Set a default file path for saving images
file_path = "captured_image.png"

# Create a folder for captured images if it doesn't exist
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Create a folder-specific file path
file_path = os.path.join("captured_images", "captured_image.png")

isRun = False

run_checkbox_key = "run_camera"
st.title("Smart App")
run = st.checkbox('Kamera starten', key=run_checkbox_key)
capture_button = st.button('Bild aufnehmen', key='capture')
FRAME_WINDOW = st.image([])

camera = PiCamera()

while run:
    if capture_button:
        camera.capture("temp_frame.jpg")  # Capture a frame and save it
        frame = Image.open("temp_frame.jpg")
    
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        # Capture and save the image
        saved_file_path = capture_and_save_image(camera, file_path)
        if saved_file_path:
            st.success(f"Bild erfolgreich aufgenommen und gespeichert: [Bild anzeigen]({saved_file_path})")
        capture_button = False

# This line will only be reached when run is False
st.write('Kamera aus')

camera.close()

# Check if the file already exists and show a link to view it
if os.path.exists(file_path):
    st.success(f"Vorheriges Bild vorhanden: [Bild anzeigen]({file_path})")
    captured_image = Image.open(file_path)
    st.image(captured_image, caption='Captured Image')
    # isolate_paper(file_path, file_path)
    pipeline_start_button = st.button('Bearbeitung starten', key='pipeline', on_click=lambda: start_run_pipeline(file_path))
