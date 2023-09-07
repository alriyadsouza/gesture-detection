from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
base_dir = "Images"
subdirectories = [chr(i) + "-samples" for i in range(ord('A'), ord('Y') + 1)]
# Load the model
model = load_model("src\keras_model.h5", compile=False)

# Load the labels
class_names = open("src\labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
for subdir in subdirectories:
    subdir_path = os.path.join(base_dir, subdir)
    
    # List all image files in the current subdirectory
    image_files = [f for f in os.listdir(subdir_path) if f.endswith(".jpg") or f.endswith(".png")]

    # Process each image in the current subdirectory
    for image_file in image_files:
        image_path = os.path.join(subdir_path, image_file)
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Predict the image
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # Print prediction and confidence score
        print("Image:", image_file)
        print("Class:", class_name[2:])
        print("Confidence Score:", confidence_score)
        print()


