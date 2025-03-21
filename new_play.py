import json
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("malayalam_sign_model_improved.h5")

# Load class mapping
with open("new_class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Reverse the class mapping to get class names from indices
class_names = {v: k for k, v in class_mapping.items()}

# Function to preprocess the input image
def preprocess_image(image, target_size=(128, 128)):
    # Resize the image to match the input size of the model
    image = cv2.resize(image, target_size)
    # Normalize the image to [0, 1]
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the class of an image
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    predictions = model.predict(processed_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    # Get the confidence score
    confidence = np.max(predictions)
    return predicted_class_name, confidence

# Test on real-time webcam input
def test_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Predict the class of the captured frame
        predicted_class, confidence = predict_image(frame)

        # Display the predicted class and confidence on the frame
        cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Real-Time Prediction", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Test on a single image file
def test_image_file(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return

    # Predict the class of the image
    predicted_class, confidence = predict_image(image)

    # Display the predicted class and confidence
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

    # Display the image with the prediction
    cv2.putText(image, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Choose the testing mode: webcam or image file
    mode = input("Enter testing mode ('webcam' or 'image'): ").strip().lower()

    if mode == "webcam":
        test_webcam()
    elif mode == "image":
        image_path = input("Enter the path to the image file: ").strip()
        test_image_file(image_path)
    else:
        print("Invalid mode. Please choose 'webcam' or 'image'.")