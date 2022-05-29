from os import write
from numpy.core.fromnumeric import argmax
from numpy.lib.type_check import imag
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import streamlit as st
import time
import base64


# streamlit interface 
st.title("Wróżenie z liścia")
st.text("Karointhegarden")
# st.text('This App classifies a flower image into Daisy/Dandelion/Rose/Sunflower/Tulip')
# For newline
st.write('\n')

image = Image.open('user_image.png')
u_img=  image.resize((299,299))
show = st.image(u_img)


st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)

    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((150,150))
    st.image(cropped_img)

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)

  
#function to process image
def process_image(img):    
    img = img.resize((224,224),Image.NEAREST)
    x = np.array(img,dtype='float32')
    x = x/255
    x = np.expand_dims(x, axis=0)
    return x 

#tflite loading the  model and getting ready for predictions
interpreter = tf.lite.Interpreter(model_path='tomato.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
output_index = output_details[0]['index']

#model prediction
def predict(X):
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds[0]

labels = ['Alternarioza', 'Bakteryjna plamistość liści', 'Septorioza', 'TMV', 'Zaraza ziemniaczana', 'Zdrowe']
label_dict = {0:'Alternarioza',1: 'Bakteryjna plamistość liści', 2:'Septorioza',3: 'TMV',4: 'Zaraza ziemniaczana',5: 'Zdrowe'}

#decode predictions
def decode_predictions(pred):
    result = {c: float(p) for c, p in zip(labels, pred)}
    result = f'Najprawdopodobniej jest to: {label_dict[pred.argmax()]}'
    return result

#main function for model prediction using tflite model and getting decoded results
def get_prediction(u_img):
    X = process_image(u_img)
    preds = predict(X)
    results = decode_predictions(preds)
    return results

# user_option = st.radio("Select an Option: ", ('Upload','URL'))
# st.write(user_option)

#take an image from user and run model prediction
st.title("Dodaj plik")
 #Give an option for uploading a file
uploaded_file = st.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )
if uploaded_file is not None:
        u_img = Image.open(uploaded_file)
        u_img = u_img.resize((224,224))
        show.image(u_img, 'Uploaded Image')
elif uploaded_file is None:        
        st.write("Załaduj plik")

#st.sidebar.button("Kliknij tutaj aby sklasyfikować")

with st.spinner('Klasyfikacja ...'):            
    prediction = get_prediction(u_img)
    time.sleep(2)
  # st.success('Done! Please check output in sidebar..')

st.header("Choroba: ")
st.success(prediction)
