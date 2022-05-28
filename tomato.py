import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Flower Classifier')

interpreter = tf.lite.Interpreter(model_path="saved_model.tflite")
interpreter.allocate_tensors()



