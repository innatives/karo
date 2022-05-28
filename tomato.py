import streamlit as st
from tflite_runtime.interpreter import Interpreter 
from PIL import Image, ImageOps
import numpy as np
import requests
import os
from io import BytesIO
import wget
import time

# def download_model():
#     model_path = 'my_model2.tflite'
#     if not os.path.exists(model_path):
#         url = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/models/my_model2.tflite'
#         filename = wget.download(url)
#     else:
#         print("Model is here.")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file inside images collections: ', filenames)
    return os.path.join(folder_path, selected_filename)


def load_labels(path): # Read the labels from the text file as a Python list.
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]



model_path = "saved_model.tflite"
label_path = "labels.txt"


def main():
    st.title("Image classification")
    image_file = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])

    if image_file != None:
        image1 = Image.open(image_file)
        rgb_im = image1.convert('RGB') 
        image = rgb_im.save("saved_image.jpg")
        image_path = "saved_image.jpg"
        st.image(image1, width = 450)

    else:
        folder_path = './images/'
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)
        image = Image.open(filename)
        image_path = filename
        print(image_path)
        st.image(image,width = 450)
        #st.image(image,use_column_width=True)

    if st.button("Make Prediction"):
        interpreter = Interpreter(model_path)
        print("Model Loaded Successfully.")

        interpreter.allocate_tensors()
        _, height, width, _ = interpreter.get_input_details()[0]['shape']
        print("Image Shape (", width, ",", height, ")")
        image = Image.open(image_path).convert('RGB').resize((width, height))

        # Run Inference.
        time1 = time.time()
        label_id, prob = classify_image(interpreter, image)
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)
        print("Classificaiton Time =", classification_time, "seconds.")

        # Read class labels.
        labels = load_labels(label_path)

        # Return the classification label of the image.
        classification_label = labels[label_id]
        print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
        st.write("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")

if __name__ == '__main__':
    main()
