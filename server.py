import cv2
import numpy as np
from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
model = load_model("trained.h5")
classes = ['Normal', 'Pneumonia']
reference_image = cv2.imread('reference_image.jpeg')  # Load your reference image here

def preprocess_image(image):
    img = cv2.resize(image, (300, 300))
    img = img / 255.0
    img = img.reshape(1, 300, 300, 3)
    return img

def compare_images(image):
    preprocessed_image = preprocess_image(image)

    # Compare the uploaded image with the reference image using SSI
    ssi_index, _ = ssim(image, reference_image, full=True)
    
    # Perform prediction only if SSI index is above a certain threshold
    if ssi_index > 0.7:
        prediction = model.predict(preprocessed_image)[0][0]
        predicted_class = classes[int(prediction >= 0.5)]
        return predicted_class, ssi_index
    else:
        return "Image is not recognized as Xray", ssi_index

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
    predicted_class, ssi_index = compare_images(image)
    return jsonify({"prediction": predicted_class, "ssi_index": ssi_index})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
