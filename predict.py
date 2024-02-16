import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_pneumonia(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (300, 300))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make prediction using the loaded model
    prediction = model.predict(img)

    # Format the prediction result
    if prediction >= 0.5:
        return "Pneumonia"
    else:
        return "Normal"

# Test the function
result = predict_pneumonia('./temp/preprocessed.jpg')
print(result)
