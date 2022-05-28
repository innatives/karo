import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
class_names = ["Bakteria", "Mozaika", "Pleśń", "Zaraza ziemniaczana"]


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


st.title('Choroby pomidorów')

## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])

if uploaded_file is not None:		
    img = Image.open(uploaded_file)
    st.image(img, caption="Input Image", width = 400)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

if st.button("Sprawdź pomidora"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)
