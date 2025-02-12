import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import Flask, request, jsonify, render_template

# Load trained model
model = load_model("potato_leaf_disease_model.h5")

# Class labels (should match dataset classes)
class_names = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

# Initialize Flask app
app = Flask(__name__)


# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure 'file' is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Check if filename is valid and allowed
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    # Ensure the 'uploads' directory exists
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the uploaded file to the 'uploads' folder
    img_path = os.path.join(upload_folder, file.filename)
    file.save(img_path)

    # Process image for prediction
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return jsonify({'class': predicted_class, 'confidence': f'{confidence}%'})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
