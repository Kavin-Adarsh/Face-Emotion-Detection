from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename

from keras.models import load_model
import cv2 # computer vision
import numpy as np 
import os
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


model = load_model('models/model_XXX.h5')

image_size = 48  
label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def predict_emotion(image_path, model, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    
    # Predict the emotion
    predictions = model.predict(image)
    
    # Get the predicted emotion label
    predicted_label = np.argmax(predictions)
    
    return predicted_label

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', filepath=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_emotion = predict_emotion(filepath, model, image_size)
        emotion = label[predicted_emotion]
        print(f"Predicted Emotion Label: {label[predicted_emotion]}")
        return render_template('index.html', filepath=filename, emotion=emotion)
    else:
        return "Invalid file type"




    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
