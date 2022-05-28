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


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])

test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)


if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)
