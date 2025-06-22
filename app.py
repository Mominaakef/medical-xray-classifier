import tensorflow as tf
import numpy as np
import os
import gdown
from tensorflow.keras.preprocessing import image
import gradio as gr

# Define model path
model_path = 'model/model.keras'

# Google Drive direct download URL (fixed!)
gdown_url = 'https://drive.google.com/uc?export=download&id=18l-Z0yz7Y71LAbc5py8-e1_6xjYChr5O'

# Download model if not present
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("🔽 Downloading model from Google Drive...")
    gdown.download(gdown_url, model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Class labels
labels = ['COVID-19', 'NORMAL', 'PNEUMONIA', 'Tuberculosis']

# Prediction function
def predict_xray(img):
    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = labels[predicted_index]
    predicted_percent = round(float(np.max(prediction)) * 100, 2)

    # Confidence scores
    confidence_scores = {labels[i]: f"{round(score * 100, 2)}%" for i, score in enumerate(prediction)}

    return f"Prediction: {predicted_class} ({predicted_percent}%)", confidence_scores

# Create Gradio interface
interface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="🩻 Medical X-ray Disease Classifier",
    description="Upload a chest X-ray image to classify it into: COVID-19, Normal, Pneumonia, or Tuberculosis."
)

# Launch the app
interface.launch()
