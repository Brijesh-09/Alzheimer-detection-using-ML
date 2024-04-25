import os
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model outside of the request handler to improve performance
model = load_model('alzheimer_detection_model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "NO Alzheimer"
    elif classNo == 1:
        return "Yes Alzheimer"
    else:
        return "Unknown"

def preprocess_image(image):
    # Add necessary preprocessing steps (resizing, normalization, etc.)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (64, 64))
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('landing.html')

@app.route('/try-now', methods=['GET'])
def try_now():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return "No file selected"
        # Save the file temporarily
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Perform prediction
        image = cv2.imread(file_path)
        if image is None:
            os.remove(file_path)
            return "Error: Unable to read image file"
        
        preprocessed_image = preprocess_image(image)
        input_img = np.expand_dims(preprocessed_image, axis=0)
        result = model.predict(input_img)
        predicted_class = get_className(np.argmax(result))
        confidence_level = float(result[0][np.argmax(result)]) * 100  # Convert to percentage
        
        # Remove the temporary file (handle permission error)
        try:
            os.remove(file_path)
        except PermissionError as e:
            print("PermissionError:", e)
        except FileNotFoundError as e:
            print("FileNotFoundError:", e)
        
        # Return the prediction result as JSON
        return jsonify({'class': predicted_class, 'confidence': confidence_level})
    return "Unknown"


if __name__ == '__main__':
    app.run(debug=True)
