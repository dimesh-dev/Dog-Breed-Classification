try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ModuleNotFoundError as e:
    import sys
    try:
        import streamlit as st
    except Exception:
        st = None

    msg = (
        "TensorFlow is not installed in this environment.\n"
        "If you're running locally, install it with:\n"
        "    pip install -r requirements.txt\n"
        "If you're deploying (Streamlit Cloud or similar), add TensorFlow to your app's requirements.\n"
        "Full error: " + str(e)
    )

    if st:
        st.error(msg)
        st.stop()
    else:
        print(msg)
        sys.exit(1)

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Dog Breed Classifier", layout="centered")

# Load Model
try:
    model = load_model(r"Image_classify.keras")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

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
    # Load image directly from memory (no temp file)
    image = Image.open(uploaded_file).convert("RGB")
    # PIL's resize expects (width, height)
    image_resized = image.resize((img_width, img_height))

    # Show uploaded image — updated to use_container_width instead of deprecated use_column_width
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = tf.keras.utils.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict with a spinner so user sees progress
    with st.spinner("Predicting..."):
        try:
            predictions = model.predict(img_array)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Get top-5 predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]

    st.subheader("Predictions:")
    for i in top_indices:
        st.write(f"**{class_names[i]}** — {predictions[0][i] * 100:.2f}%")

    # Optional: highlight top prediction
    top_idx = top_indices[0]
    st.success(f"Top prediction: **{class_names[top_idx]}** ({predictions[0][top_idx] * 100:.2f}%)")
