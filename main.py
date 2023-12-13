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




image_path_to_predict = "face/images/train/angry/0.jpg"
predicted_emotion = predict_emotion(image_path_to_predict, model, image_size)

print(f"Predicted Emotion Label: {label[predicted_emotion]}")