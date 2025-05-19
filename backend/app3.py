from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)
model = load_model("digit_recognition_model.h5")  # Ensure model is trained and present

def preprocess_digits(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((280, 280))
    img_array = np.array(img)
    gray = 255 - img_array
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_pairs = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0])
    digits = []
    for contour, (x, y, w, h) in sorted_pairs:
        if w < 5 or h < 5:
            continue
        digit_img = thresh[y:y+h, x:x+w]
        resized = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
        padded = np.pad(resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
        padded = padded.astype("float32") / 255.0
        digits.append(padded.reshape(28, 28, 1))
    return digits

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    digits = preprocess_digits(file.read())
    if len(digits) == 0:
        return jsonify({"error": "No digits found"}), 200

    digits = np.array(digits)
    preds = model.predict(digits)
    predicted = np.argmax(preds, axis=1).tolist()
    confidences = (np.max(preds, axis=1) * 100).round(2).tolist()
    return jsonify({"predicted": predicted, "confidences": confidences})

if __name__ == "__main__":
    app.run(debug=True)
