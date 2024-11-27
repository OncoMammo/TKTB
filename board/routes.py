import os
from flask import Blueprint, request, jsonify, current_app, render_template
from werkzeug.utils import secure_filename
from .utility import preprocess_image, allowed_file
from .model import model, load_testing_data
import numpy as np

# Blueprint
bp = Blueprint('routes', __name__)

# HOME
@bp.route('/')
def home():
    print("Template folder:", os.path.abspath('templates'))
    return render_template('web_html.html')

# ABOUT
@bp.route('/about')
def about():
    return render_template('about_us.html')

# SOURCE
#@bp.route('/source')
#def source():
#    return render_template('source.html')

# Go to detect
@bp.route('/predict')
def detect():
    return render_template('go_to_detect.html')

# PREDICT
# Upload, post image
@bp.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Klasifikasi citra
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        predicted_class = int(np.argmax(prediction, axis=1))

        if predicted_class == 1:
            return render_template('cancer-detected.html')
        else:
            return render_template('cancer-not-detected.html')

    return jsonify({'error': 'File not allowed'}), 400

# Additional route for testing data
@bp.route('/preedict', methods=['GET'])
def preedict():
    testing = load_testing_data()
    if testing is None:
        return jsonify({'error': 'Testing data not found or corrupted.'}), 404

    prediction = model.predict(testing)
    predicted_class = np.argmax(prediction, axis=1)
    return jsonify({'prediction': int(predicted_class[0])})
