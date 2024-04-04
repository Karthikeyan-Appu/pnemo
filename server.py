from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model("trained.h5")

def preprocess_image(image):
    img = cv2.resize(image, (300, 300))
    img = img / 255.0
    img = img.reshape(1, 300, 300, 3)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        return jsonify({"prediction": "Pneumonia" if prediction >= 0.5 else "Normal"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
