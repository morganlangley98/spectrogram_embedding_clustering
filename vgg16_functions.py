## Import required packages and model
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained VGG16 model, without the top classification layer
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract features from spectrogram images
def extract_image_features_vgg16(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # VGG16 requirese 224x224 input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features with the model
    features = model.predict(img_array)
    
    # Flatten get a 1D feature vector
    return features.flatten()