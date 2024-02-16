import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('trained.h5')

def preprocess_image(image_data):
    # Convert the base64 image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image to 300x300 pixels
    img = cv2.resize(img, (300, 300))

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Reshape the image to match the model's input shape
    img = img.reshape(1, 300, 300, 3)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.files['image'].read()

        # Preprocess the image data
        img = preprocess_image(image_data)

        # Perform prediction with the model
        prediction = model.predict(img)

        # Process the prediction result
        if prediction >= 0.5:
            result = "Pneumonia"
        else:
            result = "Normal"

        # Return the prediction result
        return jsonify({"prediction": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
