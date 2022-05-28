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
st.title("Flower Image Classification App")
st.text("Developed by Subramanian Hariharan")
st.text('This App classifies a flower image into Daisy/Dandelion/Rose/Sunflower/Tulip')
st.text('Link for reference documents are available at bottom of page')
# For newline
st.write('\n')

image = Image.open('user_image.jpg')
u_img=image.resize((299,299))
show = st.image(u_img)

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
interpreter = tf.lite.Interpreter(model_path='saved_model.tflite')
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

labels = ['daisy', 'dandelion', 'rose', 'sunflower','tulip']
label_dict = {0:'daisy',1: 'dandelion', 2:'rose',3: 'sunflower',4:'tulip'}

#decode predictions
def decode_predictions(pred):
    result = {c: float(p) for c, p in zip(labels, pred)}
    result['Prediction']=f'Given image is {label_dict[pred.argmax()]}'
    return result

#main function for model prediction using tflite model and getting decoded results
def get_prediction(u_img):
    X = process_image(u_img)
    preds = predict(X)
    results = decode_predictions(preds)
    return results

user_option = st.radio("Select an Option: ", ('Upload','URL'))
st.write(user_option)
if (user_option=='URL'):
    url = st.text_input('Enter Your Image Url(No quotes plse)')
    st.text('You have an error message if you take more than 5 sec to enter URL.')
    st.text("You may ignore error and proceed")
    time.sleep(5)
    st.write(url)    
    urllib.request.urlretrieve(url,"user_image.jpg")
    u_img = Image.open("user_image.jpg")
    u_img = u_img.resize((229,229))
    show.image(u_img, 'Uploaded Image')
elif (user_option=='Upload') :
    #take an image from user and run model prediction
    st.sidebar.title("Upload Image")
    #Give an option for uploading a file
    uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )
    if uploaded_file is not None:
        u_img = Image.open(uploaded_file)
        u_img = u_img.resize((299,299))
        show.image(u_img, 'Uploaded Image')
    elif uploaded_file is None:        
        st.sidebar.write("Please upload an Image to Classify")

#st.sidebar.button("Click Here to Classify")

with st.spinner('Classifying ...'):            
    prediction = get_prediction(u_img)
    time.sleep(2)
    st.success('Done! Please check output in sidebar..')

st.sidebar.header("Model Prediction of Probabilities and Infernce: ")
st.sidebar.write(prediction)
    
# upload pdf file with instructions
def st_display_pdf(pdf_file):
    with open(pdf_file,'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)
        
doc = st.checkbox('Display Instructions')
pdf_file_name ="SCREEN SHOTS OF TESTING.pdf"
if doc:
    st_display_pdf(pdf_file_name)

st.text(f'Flower_Classification_App_v1_{time.ctime()}')
