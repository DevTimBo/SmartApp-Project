from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Man kann sich das Bild aus dem Ordner selber aussuchen 
@app.route('/image/<filename>')
def display_image(filename):
    return send_from_directory('images\output_Images', filename)
    
if __name__ == "__main__":
    app.run(debug=True)