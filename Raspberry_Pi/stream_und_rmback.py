#### Autorin: Koudjo ####

# Importiere notwendige Module
from flask import Flask, send_file
from picamera2 import Picamera2, Preview
import time
import os
import cv2
import numpy as np

# Erstelle eine Flask-App
app = Flask(__name__)

# Definiere den Ordner, in dem hochgeladene Bilder gespeichert werden sollen
UPLOAD_FOLDER = 'input'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funktion zur Bestimmung der Reihenfolge der Punkte eines Rechtecks
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Funktion zur perspektivischen Transformation eines Bildes
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

# Definiere die Route '/capture' für das Erfassen eines Bildes
@app.route('/capture', methods=['GET'])
def capture_image():
    # Stelle sicher, dass die Datei 'test.jpg' nicht existiert, um Überschreibungsprobleme zu vermeiden
    if os.path.exists("test.jpg"):
        os.remove("test.jpg")
    
    # Initialisiere die PiCamera2
    picam2 = Picamera2()
    
    # Erstelle die Konfiguration für die Vorschau
    camera_config = picam2.create_preview_configuration()
    
    # Konfiguriere die Kamera
    picam2.configure(camera_config)
    
    # Starte die Vorschau mit der QTGL-Methode
    picam2.start_preview(Preview.QTGL)
    
    # Starte die Kamera
    picam2.start()
    
    # Warte 20 Sekunden, um sicherzustellen, dass die Kamera stabil ist
    time.sleep(20)
    
    # Erfasse das Bild und speichere es als 'test.jpg'
    picam2.capture_file("test.jpg")
    
    # Beende die Kamera
    picam2.close()
    
    # Lade das erfasste Bild
    img_path = "test.jpg"
    img = cv2.imread(img_path, 1)
    
    # Überprüfe, ob das Bild erfolgreich geladen wurde
    if img is not None:
        # Isoliere das Papier und wende die perspektivische Transformation an
        isolated_paper = isolate_paper(img)
        
        # Speichere das transformierte Bild
        cv2.imwrite("transformed.jpg", isolated_paper)
    
    # Sende die erfasste Datei als Antwort zurück
    return send_file("test.jpg", mimetype='image/jpeg')

# Funktion zum Isolieren des Papiers in einem Bild
def isolate_paper(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        paper = four_point_transform(image, screenCnt.reshape(4, 2))
        return paper
    else:
        return None

# Starte die Flask-App, wenn das Skript direkt ausgeführt wird
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
