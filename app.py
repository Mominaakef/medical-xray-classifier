from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model('model/model.keras')

# Class labels
labels = ['COVID-19', 'NORMAL', 'PNEUMONIA', 'Tuberculosis']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "Invalid file format. Please upload a PNG or JPG image.", 400

        # Save file
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = labels[predicted_index]
        predicted_percent = round(float(np.max(prediction)) * 100, 2)

        confidence_scores = {labels[i]: round(score * 100, 2) for i, score in enumerate(prediction)}

        return render_template('result.html',
                               prediction=f"{predicted_class} ({predicted_percent}%)",
                               confidence=confidence_scores,
                               image_path=filepath)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
