import os
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('Alzheimer10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0.1:
        return "Yes Alzheimer"
    elif classNo == 1.0:
        return "No Alzheimer"
    else:
        return "Yes Alzheimer"

def getResult(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result[0][0]  # Assuming your model returns a single probability

@app.route('/', methods=['GET'])
def index():
    return render_template('landing.html')

@app.route('/try-now', methods=['GET'])
def try_now():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return "No file selected"
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        os.remove(file_path)  # Delete the uploaded file after processing
        return result
    return "Unknown"

if __name__ == '__main__':
    app.run(debug=True)
