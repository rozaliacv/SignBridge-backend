from keras.models import load_model
import cv2
import numpy as np
import os
import json
import re

# Path to the test images
IMG_TEST_PATH = "test_data(60)"

# Load the same class mapping used during training
with open("class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Debugging: Print class mapping
print("\n=== Debugging Class Mapping ===")
print("Loaded class mapping:", class_mapping)
print("First few test directories:", list(os.listdir(IMG_TEST_PATH))[:5])

# Check for mismatched folders
train_classes = set(class_mapping.keys())
test_classes = set(os.listdir(IMG_TEST_PATH))
mismatched = train_classes.symmetric_difference(test_classes)
print(f"Mismatched classes: {mismatched}")

# Create reverse mapping for prediction
REV_CLASS_MAP = {int(idx): name for name, idx in class_mapping.items()}
NUM_CLASSES = len(REV_CLASS_MAP)

print("\n=== Class Mapping Debug ===")
print(f"Number of classes in model: {NUM_CLASSES}")
for idx in sorted(REV_CLASS_MAP.keys()):
    print(f"Model idx {idx} maps to class: {REV_CLASS_MAP[idx]}")

def mapper(val):
    return REV_CLASS_MAP[val]

# Create a more robust normalization function
def normalize_class_name(name):
    """Normalize class names to handle inconsistencies between training and test datasets."""
    # Convert to lowercase
    name = name.lower()
    # Remove any special characters and extra spaces
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# Create a mapping between normalized test directory names and original class names
normalized_test_to_class = {}
normalized_class_to_idx = {}

for class_name, idx in class_mapping.items():
    normalized_name = normalize_class_name(class_name)
    normalized_class_to_idx[normalized_name] = idx

for test_dir in os.listdir(IMG_TEST_PATH):
    if os.path.isdir(os.path.join(IMG_TEST_PATH, test_dir)):
        normalized_name = normalize_class_name(test_dir)
        
        # Find closest match in training classes if not exact match
        if normalized_name not in normalized_class_to_idx:
            print(f"Warning: No exact match for test directory '{test_dir}' in training classes")
            # You could add fuzzy matching here if needed
        
        normalized_test_to_class[normalized_name] = test_dir

# Print normalized mappings
print("\n=== Normalized Class Names ===")
for class_name in class_mapping.keys():
    norm = normalize_class_name(class_name)
    print(f"Original: '{class_name}' → Normalized: '{norm}'")

# Load the trained model
model = load_model("malayalam_sign_model4.h5")

# Ensure the test folder exists
if not os.path.exists(IMG_TEST_PATH):
    print(f"Error: Test folder '{IMG_TEST_PATH}' not found!")
    exit()

print("\n=== Testing Started ===\n")

# Initialize test results dictionary
test_results = {
    "total_tests": 0,
    "successful_tests": 0,
    "failed_tests": 0,
    "correct_predictions": 0,
    "predictions": []
}

for directory in os.listdir(IMG_TEST_PATH):
    path = os.path.join(IMG_TEST_PATH, directory)
    if not os.path.isdir(path):
        continue
    
    # Get normalized expected class name
    expected_class = directory
    normalized_expected = normalize_class_name(expected_class)
    
    # Check if this test directory has a matching training class
    if normalized_expected not in normalized_class_to_idx:
        print(f"Warning: Test directory '{directory}' doesn't match any training class")
        continue

    for file_name in os.listdir(path):
        # Skip hidden files and non-image files
        if file_name.startswith(".") or not (file_name.endswith(".jpg") or file_name.endswith(".png")):
            continue

        image_path = os.path.join(path, file_name)
        test_results["total_tests"] += 1

        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping {file_name}: Unable to read image")
                test_results["failed_tests"] += 1
                test_results["predictions"].append({
                    "image_path": image_path,
                    "expected_class": expected_class,
                    "status": "failed",
                    "error": "Unable to read image"
                })
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))  # Match training size
            img = img / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predict sign
            pred = model.predict(img, verbose=0)  # Suppress TF output
            sign_code = np.argmax(pred[0])
            sign_name = mapper(sign_code)
            confidence = float(pred[0][sign_code])  # Convert to native Python float

            # Compare normalized class names
            is_correct = normalize_class_name(expected_class) == normalize_class_name(sign_name)
            
            print(f" Predicted: {sign_name} (Expected: {expected_class}) for image {os.path.basename(image_path)}")
            
            # Record result
            test_results["successful_tests"] += 1
            if is_correct:
                test_results["correct_predictions"] += 1
                
            test_results["predictions"].append({
                "image_path": image_path,
                "expected_class": expected_class,
                "predicted_class": sign_name,
                "confidence": confidence,
                "status": "success",
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            test_results["failed_tests"] += 1
            test_results["predictions"].append({
                "image_path": image_path,
                "expected_class": expected_class,
                "status": "failed",
                "error": str(e)
            })

print("\n=== Testing Completed ===")

# Calculate accuracy if there were successful tests
if test_results["successful_tests"] > 0:
    accuracy = test_results["correct_predictions"] / test_results["successful_tests"]
    test_results["accuracy"] = accuracy
    print(f"Accuracy: {accuracy:.2%}")

# Save results to JSON
with open("test_results.json", "w") as f:
    json.dump(test_results, f, indent=4)

print(f"\nTest results saved to test_results.json")
print(f"Total tests: {test_results['total_tests']}")
print(f"Successful predictions: {test_results['successful_tests']}")
print(f"Correct predictions: {test_results['correct_predictions']}")
print(f"Failed tests: {test_results['failed_tests']}")

# Add confusion matrix visualization
if test_results["successful_tests"] > 0:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Extract true and predicted labels
        y_true = []
        y_pred = []
        
        for p in test_results["predictions"]:
            if p.get("status") == "success":
                y_true.append(p["expected_class"])
                y_pred.append(p["predicted_class"])
        
        # Get unique class names
        classes = sorted(list(set(y_true + y_pred)))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Visualize confusion matrix
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(xticks_rotation=90)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('images_generated/confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
    except ImportError:
        print("Could not create confusion matrix visualization (missing dependencies)")