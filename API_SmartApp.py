# TODO den Plot Part kann man cutten

import infer_pipeline
from flask import Flask, json, request, jsonify, send_from_directory
import os
import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image
from android_app.printout import output

app = Flask(__name__)
 
UPLOAD_FOLDER = 'API\images\input_Images'
input_pdf_form = r'android_app\printout\ormblatt_1.pdf'
pdf_output_path = r'API\images\PDF_output\output_form.pdf'
predictions = {}

# Maximal zulässige Dateigröße
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
@app.route('/')
def main():
    return 'Hauptseite'

# files[] ist der Schlüssel 
@app.route('/upload', methods=['POST'])
def upload_file():
    global predictions
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
 # Das request-Objekt enthält Daten, die in einer HTTP-Anfrage gesendet werden
    files = request.files.getlist('files[]')
     
    errors = {}
    success = False
     
    for file in files:      
        if file: 
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            file_path = r'API\images\input_Images\81.jpg'
            infer_pipeline.myM_prediction(file_path)
            images_with_value = infer_pipeline.myM_get_images_with_value()
            pred_texts = infer_pipeline.myM_get_pred_texts()
            for img in images_with_value:
                pred_texts = img.value 
                predictions[img.sub_class] = pred_texts
            output.fill_pdf_form(input_pdf_form, pdf_output_path, 
                                                    predictions)
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
 
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp

# Man kann sich das Bild aus dem Ordner selber aussuchen 
@app.route("/get-predictions")
def get_predictions():
    #return jsonify(predictions), 200
    directory = os.path.dirname(pdf_output_path)
    filename = os.path.basename(pdf_output_path)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(debug=True)
