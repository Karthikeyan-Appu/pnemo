from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

model = load_model("trained.h5")
classes = ['Normal', 'Pneumonia']  # Define the classes for prediction

# Load the reference image for comparison
reference_image_path = 'preprocessed_image2.jpeg'
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
reference_image = cv2.resize(reference_image, (300, 300))

def preprocess_image(image):
    img = cv2.resize(image, (300, 300))
    img = img / 255.0
    img = img.reshape(1, 300, 300, 3)
    return img

def is_chest_xray(image):
    # Calculate the Structural Similarity Index (SSIM) between the uploaded image and the reference image
    similarity_index, _ = ssim(image, reference_image, full=True)
    
    # You can adjust the threshold based on your requirements
    threshold = 0.9
    
    # If SSIM is above the threshold, consider it a chest X-ray
    if similarity_index > threshold:
        return True
    else:
        return False

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
    
    # Check if the uploaded image is a chest X-ray
    if not is_chest_xray(image):
        return jsonify({'error': 'Uploaded image is not a chest X-ray'}), 400

    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    predicted_class = classes[int(prediction >= 0.5)]
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
