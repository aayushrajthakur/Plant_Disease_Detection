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

if __name__ == "__main__":
    model = load_model('plant_disease_model.keras')
    prediction = predict_image(model, 'plant_img.jpg')

    class_labels = np.where(prediction > 0.5, 1, 0)

    # Map class labels to string values
    result_strings = ['Healthy' if label == 0 else 'Sick' for label in class_labels]

    # If the model outputs class labels directly
    # Assuming predictions are already 0 or 1
    result_strings_direct = ['Healthy' if label == 0 else 'Sick' for label in prediction]

    # Print the results
    print(result_strings)
    # print("Prediction:", prediction)
