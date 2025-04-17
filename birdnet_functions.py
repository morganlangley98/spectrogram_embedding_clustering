import tensorflow as tf
import numpy as np
import os
import cv2

#### Model related

# Path to your .tflite BirdNET model file
model_path = "/Users/morganlangley/Downloads/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



##### Preprocess the spectrogram PNG image to the correct input format


def preprocess_spectrogram(image_path, target_height=96, target_width=511):
    
    # Load the spectrogram image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to match the expected input shape (96x511)
    img_resized = cv2.resize(img, (target_width, target_height))

    # Normalize the image (scaled to [0, 1])
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Flatten the image to 1D tensor (96 * 511 = 49056), and ensure it has the correct shape
    img_flattened = img_normalized.flatten()

    # Check if we need to pad to match 144000 (if necessary, pad with zeros)
    if img_flattened.shape[0] < 144000:
        padding = 144000 - img_flattened.shape[0]
        img_flattened = np.pad(img_flattened, (0, padding), mode='constant')

    # Add batch dimension (1, 144000) to match the model's input
    img_tensor = np.expand_dims(img_flattened, axis=0)

    return img_tensor

### Extract features from the model using the processed spectrogram image


def extract_features(image_path):
    # Preprocess the spectrogram image (resize, normalize, flatten)
    img_tensor = preprocess_spectrogram(image_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_tensor)

    # Run the model inference
    interpreter.invoke()

    # Get the feature vector (output)
    feature_vector = interpreter.get_tensor(output_details[0]['index'])

    return feature_vector