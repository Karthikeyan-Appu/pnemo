from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model("trained.h5")
classes = ['Normal', 'Pneumonia']  # Define the classes for prediction

def preprocess_image(image):
    img = cv2.resize(image, (300, 300))
    img = img / 255.0
    img = img.reshape(1, 300, 300, 3)
    return img

@app.route("/")
def index():
    return "Welcome to the Pneumonia Detection API!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    predicted_class = classes[int(prediction >= 0.5)]
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
