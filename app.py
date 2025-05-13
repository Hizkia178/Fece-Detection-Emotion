from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/emotion_model.h5'
IMG_SIZE = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH)

def predict_emotion(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    label = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            label, confidence = predict_emotion(filepath)
            result = f'Prediksi: {label} ({confidence*100:.2f}%)'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
