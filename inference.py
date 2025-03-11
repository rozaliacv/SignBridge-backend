import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image

app = Flask(__name__)
# More explicit CORS configuration
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": "*"}})

class SignLanguageDetector:
    def __init__(self, model_path, class_mapping_path):
        # Load model and configuration
        print("Loading model and configuration...")
        self.model = load_model(model_path)
        
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.img_size = 128  # Updated to match training size
        
        # Add prediction history for temporal smoothing
        self.prediction_history = []
        self.history_length = 5  # Store predictions for last 5 frames
    
    def preprocess_image(self, image_data):
        # Convert base64 to image
        if isinstance(image_data, str) and ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (OpenCV uses BGR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_img, (self.img_size, self.img_size))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, image_data):
        # Preprocess image
        processed_image = self.preprocess_image(image_data)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Store prediction for temporal smoothing
        self.prediction_history.append(predictions[0])
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        # Apply temporal smoothing by averaging predictions
        if len(self.prediction_history) > 1:
            avg_predictions = np.mean(self.prediction_history, axis=0)
        else:
            avg_predictions = predictions[0]
        
        # Get top 3 predictions from smoothed results
        top_3_idx = np.argsort(avg_predictions)[-3:][::-1]
        top_3_probs = avg_predictions[top_3_idx]
        top_3_classes = [self.idx_to_class[idx] for idx in top_3_idx]
        
        return list(zip(top_3_classes, top_3_probs.tolist()))  # Convert numpy floats to Python floats

# Initialize detector
detector = SignLanguageDetector(
    model_path="malayalam_sign_model.h5",  # Using the best model
    class_mapping_path="class_mapping.json"
)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
        
    try:
        image_data = request.json['image']
        print("Received image data:", image_data[:50] + "...")  # Log first 50 chars of base64 image
        
        # Decode and save the image for debugging
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image.save("debug_image.jpg")  # Save the image to check if it's valid
        print("About to call detector.predict...")
        predictions = detector.predict(image_data)
        print("Predictions:", predictions)  # Log predictions
        
        # Format predictions
        formatted_predictions = [
            {'class': cls, 'confidence': float(conf)} 
            for cls, conf in predictions
        ]
        
        return jsonify({
            'predictions': formatted_predictions,
            'timestamp': time.time()
        })
    except Exception as e:
        print("Error:", str(e))  # Log any errors
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)