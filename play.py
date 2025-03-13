import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
import time

class SignLanguageDetector:
    def __init__(self, model_path, class_mapping_path):
        # Load model and configuration
        print("Loading model and configuration...")
        self.model = load_model(model_path)
        
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.img_size = 128
        
        # Initialize performance metrics
        self.inference_times = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Add prediction history for temporal smoothing
        self.prediction_history = []
        self.history_length = 5  # Store predictions for last 5 frames
    
    def preprocess_frame(self, frame):
        # Convert to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self.img_size, self.img_size))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, frame):
        # Start timing
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        
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
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return list(zip(top_3_classes, top_3_probs)), inference_time
    
    def draw_predictions(self, frame, predictions, inference_time):
        # Draw prediction boxes and labels
        for i, (class_name, prob) in enumerate(predictions):
            # Draw prediction text
            text = f"{class_name}: {prob:.1%}"
            cv2.putText(frame, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw inference time
        fps_text = f"FPS: {1/np.mean(self.inference_times[-100:]):.1f}"
        cv2.putText(frame, fps_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame

def main():
    # Initialize detector
    detector = SignLanguageDetector(
        model_path="malayalam_sign_model.h5",
        class_mapping_path="class_mapping.json"
    )
    
    # Initialize veo capture
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Make prediction
            predictions, inference_time = detector.predict(frame)
            
            # Draw predictions on frame
            frame = detector.draw_predictions(frame, predictions, inference_time)
            
            # Show frame
            cv2.imshow('Sign Language Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        avg_fps = len(detector.inference_times) / (time.time() - detector.start_time)
        avg_inference = np.mean(detector.inference_times) * 1000
        print(f"\nPerformance Statistics:")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average inference time: {avg_inference:.1f}ms")

if __name__ == "__main__":
    main()