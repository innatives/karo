import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./flower_model_trained.hdf5')
	return model


#load image
@st.cache(show_spinner=False)
def load_img(path):
        ## reading file object and making it to pil image and to np array
        img_l=[]
        for i in path:
                img_byte=i.read()
                img=Image.open(io.BytesIO(img_byte))
                img=img.resize((256,256),Image.ANTIALIAS)
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

def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction

st.title('Flower Classifier')

inp_t = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])
model = load_model()

if inp_t is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(inp_t)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)

