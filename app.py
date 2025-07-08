import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('🌼 Flower Classification CNN Model') 
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the model once
model = load_model('Flower_Recog_Model.keras')

def classify_images(image_file):
    input_image = tf.keras.utils.load_img(image_file, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = '🌸 The image belongs to **' + flower_names[np.argmax(result)] + '** with a score of **' + str(np.max(result) * 100)[:5] + '%**.'
    return outcome

uploaded_file = st.file_uploader('📤 Upload an image of a flower', type=['jpg', 'png', 'jpeg'])

# Proceed only if a file is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, width=250, caption='Uploaded Image')
    
    # Classify the image and display result
    result_text = classify_images(uploaded_file)
    st.markdown(result_text)
