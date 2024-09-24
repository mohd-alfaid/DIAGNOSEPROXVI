import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the ResNet50 model
model = ResNet50(weights='imagenet')

# Set up the Streamlit app
st.title("Breast Cancer Detection App")
st.write("Upload a medical image for prediction.")

# File uploader for image uploads
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image and display it
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=3)[0]

    # Display the predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        st.write(f"{i + 1}: {label} - {score:.2f}")

