from __future__ import division, print_function
import numpy as np
import os
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

import sys
import glob
import re

app = Flask(__name__)

model_path = 'Xception1.h5'
model = load_model(model_path)

num_classes = model.output_shape[-1]

class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        filepath = os.path.join(upload_path, secure_filename(f.filename))
        f.save(filepath)
        
        img = image.load_img(filepath, target_size=(299, 299))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='caffe')
        preds = model.predict(x)
        predicted_class = np.argmax(preds, axis=1)[0]
        
        result = class_names[predicted_class]
        
        return jsonify(result=result)
    return "Invalid request method."

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
