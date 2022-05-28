import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)

tflite_interpreter = tf.lite.Interpreter(model_path='saved_model.tflite')
tflite_interpreter.allocate_tensors()

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_predictions(input_image):
    output_details = tflite_interpreter.get_output_details()
    set_input_tensor(tflite_interpreter, input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    pred_class = class_names[tflite_model_prediction]
    return pred_class

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./flower_model_trained.hdf5')
	return model


st.title('Flower Classifier')

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])

## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(224,224,3), interpolation='nearest')
    st.image(img)
    print(value)
    if value == 2 or value == 5:
        img = tf.image.convert_image_dtype(img, tf.uint8)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)


if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)
