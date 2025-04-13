# classifier/utils.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model

MODEL_PATH = "classifier/models/empty_image_detector.h5"
model = load_model(MODEL_PATH)

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    is_empty = bool(prediction[0][0] > 0.5)  # Assuming binary classification
    return is_empty