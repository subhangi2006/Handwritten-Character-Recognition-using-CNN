"""Streamlit app for handwritten digit recognition. Train first with python train.py."""
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "handwritten_digit_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Recognition",
                   page_icon="✍️", layout="wide")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image of one handwritten digit (0–9). A CNN trained on "
         "the MNIST dataset will predict the digit.")
st.info("For best results, upload a clear image containing one centered digit.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image):
    """Convert an uploaded image to a centered 28x28 MNIST-style image."""
    image = ImageOps.grayscale(image)
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    image = image.point(lambda pixel: 255 if pixel > 40 else 0)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(image, ((28 - image.width) // 2,
                         (28 - image.height) // 2))
    array = np.asarray(canvas, dtype="float32") / 255.0
    return array.reshape(1, 28, 28, 1), canvas


if not MODEL_PATH.exists():
    st.warning("The model is not trained yet. In the project terminal, run "
               "python train.py. After training finishes, refresh this page.")
    st.stop()

model = load_model()
uploaded_file = st.file_uploader(
    "Upload a digit image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file:
    try:
        original = Image.open(uploaded_file).convert("RGB")
        input_image, processed = prepare_image(original)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded image")
            st.image(original, use_container_width=True)
        with col2:
            st.subheader("Processed 28 × 28 image")
            st.image(processed.resize((280, 280)),
                     caption="Image passed to the CNN")

        probabilities = model.predict(input_image, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit]) * 100

        st.success(f"Predicted digit: {digit}")
        st.metric("Prediction confidence", f"{confidence:.2f}%")
        st.subheader("Class probabilities")
        st.bar_chart({str(i): float(probabilities[i]) for i in range(10)})
        st.caption("Confidence is the model's predicted probability, not a "
                   "guarantee that the prediction is correct.")
    except Exception as error:
        st.error(f"Could not process this image: {error}")
else:
    st.markdown("### How to use")
    st.markdown("1. Upload a clear image of one handwritten digit.\n"
                "2. Review the processed image.\n"
                "3. View the predicted digit and class probabilities.")

st.divider()
st.caption("Built with Python, TensorFlow/Keras, MNIST and Streamlit.")
