import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.title('Tomato')
        
class_labels={0:'Bakteria',1:'Mozaika',2:'Pleśń',3:'Zaraza ziemniaczana'}

inp_t = st.file_uploader(label='Upload',accept_multiple_files=True)

#load image
@st.cache(show_spinner=False)
def load_img(path):
        ## reading file object and making it to pil image and to np array
        img_l=[]
        for i in path:
                img_byte=i.read()
                img=Image.open(io.BytesIO(img_byte))
                img=img.resize((224,224),Image.ANTIALIAS)
                if img.mode!='L':
                        img=img.convert('L')
                img_arr=np.array(img,dtype='float32')/255
                img_arr=np.expand_dims(img_arr,axis=-1)
                img_l.append(img_arr)
        img=np.stack(img_l)
        return img

## prediction
@st.cache(show_spinner=False)
def pred(img):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path = r'saved_model.tflite')

    # setting input size
    interpreter.resize_tensor_input(0, [img.shape[0],224,224,1], strict=True)
    interpreter.allocate_tensors()
    #interpreter = load_model()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]['shape']

    input_data = img
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    # making predictions
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    #tflite_resluts = (tflite_results)

    tflite_results = np.concatenate(tflite_results)  # merging all sub arrays
    
    tf_results=[1 if i>0.5 else 0 for i in tflite_results]  # scaling predictions to 0,1.
 
    return tflite_results,tf_results

vis_img = st.sidebar.checkbox('Show Uploaded Images')



## prints model arch flow chart
#if st.sidebar.checkbox('Model Architecture'):            
#        st.sidebar.write(model)
