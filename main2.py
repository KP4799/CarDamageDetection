import streamlit as st
import tensorflow as tf
#from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# hack to change model config from keras 2->3 compliant
import h5py

f = h5py.File("keras_model.h5", mode="r+")
model_config_string = f.attrs.get("model_config")
if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
    model_config_string = f.attrs.get("model_config")
    assert model_config_string.find('"groups": 1,') == -1

f.close()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1

st.title("Car Damage Detection")
file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def import_and_predict(image_path,model):
    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Print prediction and confidence score
    st.write("Class: {}".format(class_name[2:]))
    st.write("Confidence Score: {}".format(confidence_score))

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

if file is None:
    st.write("No image uploaded")
else:
    st.image(file, use_column_width=True)
    import_and_predict(file, model)
