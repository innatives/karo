import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Flower Classifier')

interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
input_details

