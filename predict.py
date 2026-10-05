import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model("model/waste_classifier.keras")

# Waste classes
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Image to predict
image_path = "test_image.jpg"

# Load and prepare image
image = load_img(image_path, target_size=(128, 128))
image = img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make prediction
prediction = model.predict(image, verbose=0)

predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Predicted waste:", predicted_class)
print(f"Confidence: {confidence:.2f}%")