from flask import Flask, request, render_template, jsonify
from utils import load_model, predict
import os

app = Flask(__name__)
model = load_model("model/apple2.pth")  # path to your model file
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/log')
def log():
    return render_template("login.html")

@app.route('/solutions')
def sol():
    return render_template("solutions.html")

@app.route('/about')
def about():
    return render_template("aboutUs.html")    
@app.route('/reg')
def reg():
    return render_template("register.html")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/identify')
def identify():
    return render_template('identifier.html')  # your frontend HTML

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    prediction = predict(model, filepath)
    os.remove(filepath)  # clean up

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
