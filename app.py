import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

class_names = ["Bakteria", "Mozaika", "Pleśń", "Zaraza ziemniaczana"]


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Choroby pomidorów')
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])


tflite_interpreter = tf.lite.Interpreter(model_path='saved_model.tflite')
input_details = tflite_interpreter.get_input_details()
req_input_size = (2, 224, 224, 1) #Your input size
tflite_interpreter.resize_tensor_input(input_details[0]['index'], req_input_size)
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


## Input Fields


if uploaded_file is not None:	
    img = tf.keras.preprocessing.image.load_img(uploaded_file , grayscale=False, color_mode='rgb', target_size=(224,224), interpolation='nearest')
    img_array = tf.keras.utils.img_to_array(img)	
    img_array = tf.expand_dims(img_array, 0)
    st.image(img, caption="Input Image", width = 400)

if st.button("Sprawdź pomidora"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)
