const express = require('express');
const multer = require('multer');
const cv = require('opencv4nodejs');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const { PythonShell } = require('python-shell');

const app = express();
const PORT = process.env.PORT || 3000;

// Configure multer to handle file uploads
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, './uploads/');
  },
  filename: function(req, file, cb) {
    cb(null, file.originalname);
  }
});

const upload = multer({ storage: storage });

// Load the trained model
const { spawn } = require('child_process');
const model = spawn('python', ['load_model.py']);

// Route for uploading image
app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    // Load the image using OpenCV
    const img = cv.imread(`./uploads/${req.file.filename}`);

    // Preprocess the image
    const preprocessedImg = preprocessImage(img);

    // Save the preprocessed image temporarily
    cv.imwrite('./temp/preprocessed.jpg', preprocessedImg);

    // Call Python script to make predictions
    PythonShell.run('predict.py', null, function (err, result) {
      if (err) throw err;
      // Send back the prediction result
      res.json({ prediction: result });
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Preprocessing function
function preprocessImage(img) {
  // Resize image to match model input size
  const resizedImg = img.resize(300, 300);
  // Convert image to grayscale
  const grayImg = resizedImg.bgrToGray();
  // Normalize pixel values
  const normalizedImg = grayImg.normalize(0, 255);
  // Expand dimensions to match model input shape
  const expandedImg = normalizedImg.resize(1, 300, 300);
  return expandedImg;
}

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
