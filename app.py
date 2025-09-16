try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ModuleNotFoundError as e:
    import sys
    import streamlit as st

    st = st if 'st' in globals() else None
    msg = (
        "TensorFlow is not installed in this environment.\n"
        "If you're running locally, install it with:\n"
        "    pip install -r requirements.txt\n"
        "If you're deploying (Streamlit Cloud or similar), add TensorFlow to your app's requirements.\n"
        "Full error: " + str(e)
    )

    # If Streamlit is available show the message in-app, otherwise print and exit
    if st:
        st.error(msg)
        st.stop()
    else:
        print(msg)
        sys.exit(1)

import streamlit as st
import numpy as np
import tempfile
import os

# Load Model

model = load_model(r"C:\Users\Dimesh Prasantha\OneDrive\Documents\Projects\Dog-Breed-Classification\Image_classify.keras")

# Dog breed classes
class_names = [
    'afghan_hound', 'african_hunting_dog', 'airedale', 'basenji', 'basset', 'beagle',
    'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound',
    'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier',
    'borzoi', 'boston_bull', 'bouvier_des_flandres', 'brabancon_griffon', 'bull_mastiff',
    'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chow', 'clumber', 'cocker_spaniel',
    'collie', 'curly-coated_retriever', 'dhole', 'dingo', 'doberman', 'english_foxhound',
    'english_setter', 'entlebucher', 'flat-coated_retriever', 'german_shepherd',
    'german_short-haired_pointer', 'golden_retriever', 'gordon_setter', 'great_dane',
    'great_pyrenees', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier',
    'irish_water_spaniel', 'irish_wolfhound', 'japanese_spaniel', 'keeshond',
    'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'leonberg', 'lhasa',
    'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher',
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese',
    'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
    'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier',
    'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'standard_poodle',
    'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier',
    'toy_terrier', 'vizsla', 'weimaraner', 'whippet', 'wire-haired_fox_terrier',
    'yorkshire_terrier'
]

img_height, img_width = 224, 224


# UI

st.title("🐶 Dog Breed Classification App")
st.write("Upload a dog image and I will predict its breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    img_path = temp_file.name

    # Show image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_load = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image_load)
    img_array = tf.expand_dims(img_array, 0) / 255.0  

    # Predict
    predictions = model.predict(img_array)
    top_indices = predictions[0].argsort()[-5:][::-1] 

    st.subheader("Predictions:")
    for i in top_indices:
        st.write(f"**{class_names[i]}**: {predictions[0][i] * 100:.2f}%")

    # Cleanup temp file
    os.unlink(img_path)
