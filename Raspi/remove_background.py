import cv2
import numpy as np
import os

def remove_background(imagePath):
    script_directory = os.path.dirname(os.path.realpath(__file__))
     
    image_dir = os.path.join(script_directory, imagePath)
    # Bild einlesen
    image = cv2.imread(image_dir)
    
   
    
    # Konvertiere das Bild zu Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Wende einen Schwellenwert an, um Vordergrund und Hintergrund zu trennen
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Finde Konturen im Bild
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Erstelle eine Maske für den Vordergrund
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Extrahiere den Vordergrund
    result = cv2.bitwise_and(image, mask)
    
    # Zeige das Originalbild und das Ergebnis an
    cv2.imshow('Original', image)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Beispielaufruf
remove_background("dein-bild.jpeg")
