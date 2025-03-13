import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.utils import class_weight
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
from multiprocessing import freeze_support


# Functions and model definition
def enable_gpu():
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled GPU memory growth")
        except RuntimeError as e:
            print(e)

def get_model(image_size, num_classes):
    base_model = MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights="imagenet")
    
    # Unfreeze last few layers for faster adaptation
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.6),  # Increased dropout
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Reduced neurons
        layers.BatchNormalization(),
        layers.Dropout(0.6),  # Increased dropout
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

def train_model():
    enable_gpu()
    set_global_policy('mixed_float16')

    DATASET_PATH = "image_data"
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32

    os.makedirs("images_generated", exist_ok=True)

    class_names = sorted(os.listdir(DATASET_PATH))
    NUM_CLASSES = len(class_names)
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    with open("class_mapping.json", "w") as f:
        json.dump(class_to_index, f)
    print("Saved class mapping to class_mapping.json")

    class_counts = {}
    for class_name in class_names:
        path = os.path.join(DATASET_PATH, class_name)
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        class_counts[class_name] = count
        
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    print(f"Class imbalance ratio: {max_samples/min_samples:.2f}x")

    # More aggressive data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,  # Increased rotation
        width_shift_range=0.2,  # Increased shift
        height_shift_range=0.2,  # Increased shift
        zoom_range=0.2,  # Increased zoom
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Wider brightness range
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_generator = valid_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    model = get_model(IMAGE_SIZE, NUM_CLASSES)

    

#

# Compile the model with the learning rate schedule
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),  # Pass the schedule here
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, min_delta=0.01),  # Reduced patience
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    print("Training the model (optimized for speed)...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,  # Increased epochs
        callbacks=callbacks,
        class_weight=class_weights_dict,
        workers=1,
        use_multiprocessing=False
    )

    model.save("malayalam_sign_model3.h5")
    print("Model saved as malayalam_sign_model3.h5")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("images_generated/training_history3.png")
    plt.close()
    print("Training history plot saved to images_generated/training_history.png")
    print("\nEvaluating final model performance...")
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    test_generator = valid_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    evaluation = model.evaluate(test_generator)
    print(f"\nTest Loss: {evaluation[0]:.4f}")
    print(f"Test Accuracy: {evaluation[1]:.4f}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("images_generated/confusion_matrix.png")
    plt.close()

    with open("model_metrics.json", "w") as f:
        json.dump({
            "test_loss": float(evaluation[0]),
            "test_accuracy": float(evaluation[1]),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE
        }, f)

    print("Final model evaluation complete!")

if __name__ == "__main__":
    freeze_support()
    train_model()