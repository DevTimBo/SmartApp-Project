import pipeline
from flask import Flask, json, request, jsonify, send_from_directory
import os
import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
 
UPLOAD_FOLDER = 'API\images\input_Images'
DOWNLOAD_FOLDER = 'API\images\output_Images'
loaded_model = pipeline.load_model_and_weights()
#test_image = r'handwriting\data\a01-000u-00.png'

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
            #image_rotate(filename)
            #file_path = os.join.path(UPLOAD_FOLDER, filename)
            file_path = r'API\images\input_Images\123.png'
            prediction = pipeline.infer(loaded_model, file_path)
            print(prediction)
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
@app.route("/get-prediction/<pid>")
def get_user(pid):
    user_data = {
        "pid": pid,
        "name": "John",
        "prediction": prediction
    }

    extra = request.args.get("extra")
    
    if extra:
        user_data["extra"] = extra

    return jsonify(user_data), 200

""" def image_rotate(filename): 
    myImage = Image.open(f"API\images\input_Images\{filename}")
    rotated_image = myImage.rotate(90, expand=True)
    #rotated_image.show()
    rotated_image.save(os.path.join(DOWNLOAD_FOLDER, filename))  """

 
if __name__ == '__main__':
    app.run(debug=True)

#pipeline.infer(loaded_model, test_image)