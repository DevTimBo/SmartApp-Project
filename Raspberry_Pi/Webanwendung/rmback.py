#### Autorin: Koudjo ####

# Importieren der benötigten Bibliotheken
import cv2
import numpy as np

def order_points(pts):
    # Sortiere die Punkte im Uhrzeigersinn, beginnend beim oberen linken Punkt
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # Erhalte eine konsistente Reihenfolge der Punkte und entpacke sie einzeln
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Berechne die Breite des neuen Bildes als maximale Entfernung zwischen bottom-right und bottom-left
    # x-Koordinaten oder top-right und top-left x-Koordinaten
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Berechne die Höhe des neuen Bildes als maximale Entfernung zwischen top-right und bottom-right
    # y-Koordinaten oder top-left und bottom-left y-Koordinaten
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Jetzt, da wir die Dimensionen des neuen Bildes haben, erstelle den Satz von Zielpunkten, um eine "Vogelperspektive" zu erhalten
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Berechne die Perspektiven-Transformationsmatrix und wende sie an
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Gib das transformierte Bild zurück
    return warped

def isolate_paper(uploaded_image):
    # Konvertiere die hochgeladene Datei in das OpenCV-Bildformat, falls es noch keine Bildmatrix ist
    if not isinstance(uploaded_image, np.ndarray):
        # Annahme: uploaded_image ist ein dateiähnliches Objekt
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        image = uploaded_image

    # Konvertiere in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Kantenerkennung
    edged = cv2.Canny(blur, 75, 200)

    # Finde Konturen und sortiere sie nach Größe absteigend
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # Iteriere über die Konturen
    for c in contours:
        # Nähere die Kontur an
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Falls unsere approximierte Kontur vier Punkte hat, nehmen wir an, dass wir das Papier gefunden haben
        if len(approx) == 4:
            screenCnt = approx
            break

    # Falls eine rechteckige Kontur gefunden wurde, wende die perspektivische Transformation an
    if screenCnt is not None:
        paper = four_point_transform(image, screenCnt.reshape(4, 2))
        return paper
    else:
        # Falls keine rechteckige Kontur gefunden wurde, gib None oder das Originalbild zurück
        return None
