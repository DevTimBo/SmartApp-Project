# TODO 

import infer_pipeline
from flask import Flask, json, request, jsonify, send_from_directory
import os
import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
 
UPLOAD_FOLDER = 'API\images\input_Images'
DOWNLOAD_FOLDER = 'API\images\output_Images'

# Maximal zulässige Dateigröße
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
@app.route('/')
def main():
    return 'Hauptseite'

# files[] ist der Schlüssel 
@app.route('/upload', methods=['POST'])
def upload_file():
    global prediction
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
            #image_rotate(filename)
            #file_path = os.join.path(UPLOAD_FOLDER, filename)
            file_path = r'API\images\input_Images\81.jpg'
            #prediction = pipeline.infer_with_return(loaded_model, file_path)
            #print(prediction)
            infer_pipeline.myM_prediction(file_path)
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

if __name__ == '__main__':
    app.run(debug=True)
