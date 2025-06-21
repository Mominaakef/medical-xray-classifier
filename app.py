from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# Initialize Flask app
app = Flask(__name__)

# Path to local model
model_path = 'model/model.keras'

# Google Drive model file ID
drive_id = '18l-Z0yz7Y71LAbc5py8-e1_6xjYChr5O'
gdown_url = f'https://drive.google.com/uc?id={drive_id}'

# Download model from Drive if not already present
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("🔽 Downloading model from Google Drive...")
    gdown.download(gdown_url, model_path, quiet=False)

# Load the trained model
model = load_model(model_path)

# Class labels (make sure order matches training)
labels = ['COVID-19', 'NORMAL', 'PNEUMONIA', 'Tuberculosis']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image to static folder
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = labels[predicted_index]
    predicted_percent = round(float(np.max(prediction)) * 100, 2)

    # Class-wise confidence
    confidence_scores = {labels[i]: round(score * 100, 2) for i, score in enumerate(prediction)}

    # Return result to result.html
    return render_template('result.html',
                           prediction=f"{predicted_class} ({predicted_percent}%)",
                           confidence=confidence_scores,
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
