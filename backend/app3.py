import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS support
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and compile model
model = load_model("digit_recognition_model.h5")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def preprocess_digits(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((280, 280))
        img_array = np.array(img)
        gray = 255 - img_array
        
        # Thresholding and morphology
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return []

        # Process each digit
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_pairs = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0])
        
        digits = []
        for contour, (x, y, w, h) in sorted_pairs:
            if w < 5 or h < 5:  # Skip small artifacts
                continue
                
            digit_img = thresh[y:y+h, x:x+w]
            resized = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
            padded = np.pad(resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
            digits.append((padded / 255.0).reshape(28, 28, 1))
            
        return digits
        
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return []

@app.route('/')
def home():
    return "Digit Recognition API is running", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400
            
        # Read and validate image
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty image data"}), 400
            
        # Process image
        digits = preprocess_digits(img_bytes)
        if len(digits) == 0:
            return jsonify({"error": "No digits detected"}), 200
            
        # Make predictions
        digits_array = np.array(digits)
        preds = model.predict(digits_array)
        
        return jsonify({
            "predicted": np.argmax(preds, axis=1).tolist(),
            "confidences": (np.max(preds, axis=1) * 100).round(2).tolist()
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
