#### Autorin: Koudjo ####

# Importieren der benötigten Module und Funktionen
from inferenz_pipeline import run_pipeline
from inferenz_pipeline import get_images_with_value
from inferenz_pipeline import get_pred_texts
from flask import Flask, jsonify, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
from simpleTimer import SimpleTimer
import time
import numpy as np
from flask import send_file
from android_app.printout import output

# Initialisierung der Flask-App
app = Flask(__name__)

# Konfiguration der Upload- und Download-Verzeichnisse
UPLOAD_FOLDER = 'PI_API/images/input_Images'
DOWNLOAD_FOLDER = 'PI_API/images/output_Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Maximale zulässige Dateigröße
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialisierung eines einfachen Timers
timer = SimpleTimer()

# Routen-Definitionen

# Hauptseite
@app.route('/')
def main():
    return 'Hauptseite'

# Datei-Upload-Route
@app.route('/upload', methods=['POST'])
def upload_file():
    # Sicherstellen, dass die Verzeichnisse existieren
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

    # Überprüfen, ob Dateien im POST-Request enthalten sind
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    # Dateien aus dem Request abrufen
    files = request.files.getlist('files[]')
    errors = {}
    success = False
    elapsed_time = 0

    for file in files:
        if file:
            # Dateiname sichern und Pfad erstellen
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Datei speichern
            file.save(filepath)
            print(filepath)
            
            # Bildverarbeitungspipeline ausführen
            timer.start()
            run_pipeline(filepath)
            timer.stop()
            print("Pipeline over")
            
            # Ergebnisse der Bildverarbeitung abrufen
            images_with_value = get_images_with_value()
            pred_texts = get_pred_texts()
            
            # Vorhersagen in einem Dictionary speichern
            for img in images_with_value:
                pred_texts = img.value 
                predictions[img.sub_class] = pred_texts
                output.fill_pdf_form(input_pdf_form, pdf_output_path, 
                                                    predictions) 
            
            # Gesamtzeit für die Verarbeitung speichern
            elapsed_time = timer.get_elapsed_time()
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'

    # Antwort je nach Erfolg oder Fehler erstellen
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message': 'Files successfully uploaded', 'time': {elapsed_time}})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp

# PDF-Download-Route
@app.route('/downloads', methods=['GET'])
def download_output_pdf():
    # PDF-Ausgabepfad
    pdf_output_path = 'PI_API/pdf_output/output_form.pdf'
    headers = {'Content-Type': 'application/pdf'}
    
    # Überprüfen, ob die Datei existiert
    if os.path.exists(pdf_output_path):
        # Datei als Anhang für den Download senden
        return send_file(pdf_output_path, as_attachment=True), 200, headers
    else:
        return 'File not found', 404

# Anzeige von Bildern aus dem Download-Verzeichnis
@app.route('/image/<filename>')
def display_image(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)

# Funktion zum Drehen von Bildern
def image_rotate(filename):
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    myImage = Image.open(input_path)
    rotated_image = myImage.rotate(90, expand=True)
    rotated_image.save(output_path)

# Start der Flask-App
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
