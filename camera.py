import cv2
import streamlit as st

st.title("Webcam Live Feed")
run = st.checkbox('Run')
capture_button = st.button('Bild aufnehmen')
pipeline_start_button = st.button('Bearbeitung starten')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Kamera aus')