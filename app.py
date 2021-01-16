import streamlit as st
import tensorflow as tf
from io import BytesIO
import requests
import cv2
from PIL import Image, ImageOps
import numpy as np


def main():
  html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h3 style="color:white;text-align:center;">Waste Classification with Transfer Learning Models </h3>
    </div>
    """
  st.markdown(html_temp, unsafe_allow_html=True)
if __name__ == '__main__':
  main()










st.page_title="Image Classification App",
page_icon="✨",
layout="wide",
initial_sidebar_state="expanded"
st.sidebar.subheader("Input")
st.sidebar.markdown("Welcome to the Waste Classification App")
page = st.sidebar.selectbox("Select the Model", [ "EfficientNetB7" , "MobileNetV2", "Xception"])   # pages



if page == "EfficientNetB7":
  st.set_option('deprecation.showfileUploaderEncoding', False)

  @st.cache(allow_output_mutation = True)
  def load_model():
    model = tf.keras.models.load_model('EfficientNetB0.h5')
    return model
  with st.spinner('Loading Model Into Memory ....'):
    model = load_model()

  file = st.file_uploader("", type=["jpg", "jpeg", "png"])

  def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


  if file is None:
    st.warning("Please upload an image file")
  else:
    image= Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image,model)
    class_names= [ 'Cardboard', 'E-Waste','Glass', 'Metal','Paper', 'Plastic', 'Trash']
    string = "This image most likely is : "+class_names[np.argmax(predictions)]+ " category according to the EfficientNetB7 Model"
    st.success(string)
    image = Image.open('2.jpeg')
    st.image(image, caption='Confusion Matrix of EfficientNetB7 Model',use_column_width=True)



elif page == "MobileNetV2":
  st.set_option('deprecation.showfileUploaderEncoding', False)


  @st.cache(allow_output_mutation=True)
  def load_model():
    model = tf.keras.models.load_model('my_model.hdf5')  # my_model.hdf5
    return model
  model = load_model()


  file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

  def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

  if file is None:
    st.warning(" Please upload an image file")
  else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Cardboard', 'E-Waste', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    string = "This image most likely is : " + class_names[
      np.argmax(predictions)] + " category according to the MobileNetV2 Model"
    st.success(string)
    image = Image.open('3.png')
    st.image(image, caption='Confusion Matrix of MobileNetV2 Model', use_column_width=True)



elif page == "Xception":
  st.set_option('deprecation.showfileUploaderEncoding', False)
  @st.cache(allow_output_mutation=True)
  def load_model():
    model = tf.keras.models.load_model('Xception93.h5')  
    return model
  model = load_model()
  file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])
  import cv2
  from PIL import Image, ImageOps
  import numpy as np

  def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

  if file is None:
    st.warning("Please upload an image file")

  else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Cardboard', 'E-Waste', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    string = "This image most likely is : " + class_names[
      np.argmax(predictions)] + " category according to the Xception Model"
    st.success(string)
    image = Image.open('4.png')
    st.image(image, caption='Confusion Matrix of Xception Model', use_column_width=True)



#image = Image.open('1.jpg')
#st.image(image, caption='Our Published Journal Based on FYP1 Result',use_column_width=True)














