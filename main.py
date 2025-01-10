from urllib import request
import os

import numpy as np
import tensorflow as tf
import predict
from werkzeug.utils import secure_filename
from predict import predict_image  # Import the predict_image function
from PIL import Image
from flask import Flask,app, redirect, url_for, request, render_template, flash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files.get('image')

        if image is None or image.filename == '':
            flash('No selected file!', 'danger')
            return redirect(url_for('/'))
        
        filename = secure_filename(image.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(img_path)
        image = Image.open(img_path)
        model = load_model('plant_disease_model.keras')
        prediction = predict_image(model, img_path) 

        class_labels = np.where(prediction > 0.5, 1, 0)

        
        res = ['Healthy' if label == 0 else 'Sick' for label in class_labels]

        
        result_strings_direct = ['Healthy' if label == 0 else 'Sick' for label in prediction]
        
        return render_template('output.html', image=img_path, res = res)
        
    return render_template('home.html') 

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
