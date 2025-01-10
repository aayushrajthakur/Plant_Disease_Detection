import tensorflow as tf
import numpy as np
import cv2

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)
    return prediction
