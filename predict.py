import tensorflow as tf
import numpy as np
import cv2

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction

if __name__ == "__main__":
    model = load_model('plant_disease_model.keras')
    prediction = predict_image(model, 'plant_img.jpg')
    print("Prediction:", prediction)
