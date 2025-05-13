import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Konstanta
IMG_SIZE = 48
MODEL_PATH = 'models/emotion_model.h5'
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load model
model = load_model(MODEL_PATH)

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buka webcam
cap = cv2.VideoCapture(0)

# Set resolusi kamera (opsional, tergantung kamera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Buat jendela fullscreen
cv2.namedWindow('Emotion Detection - Webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Detection - Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_norm = face_resized / 255.0
        face_input = face_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        prediction = model.predict(face_input)
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Gambar kotak & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Emotion Detection - Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
