from huggingface_hub import hf_hub_download
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Dog Breed Classifier", layout="centered")

# 🔹 Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="dimeshanthoney/dog-breed-classifier",
    filename="Image_classify.keras",
    token=os.environ.get("HF_TOKEN")  # Optional: needed if repo is private
)

# Load model
model = load_model(model_path)

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
st.write("Upload a dog image and I'll predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image directly from memory
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((img_width, img_height))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = tf.keras.utils.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)

    # Get top-5 predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]

    st.subheader("Predictions:")
    for i in top_indices:
        st.write(f"**{class_names[i]}** — {predictions[0][i] * 100:.2f}%")

    # Highlight top prediction
    top_idx = top_indices[0]
    st.success(f"Top prediction: **{class_names[top_idx]}** ({predictions[0][top_idx] * 100:.2f}%)")
