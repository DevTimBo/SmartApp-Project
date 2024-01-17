from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
from bounding_box import ressize
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'images'

# Maximal zulässige Dateigröße
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
@app.route('/')
def main():
    return 'Hauptseite'
 
# files[] ist der Schlüssel 
@app.route('/upload', methods=['POST'])
def upload_file():
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