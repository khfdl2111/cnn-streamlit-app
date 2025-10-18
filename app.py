# app.py
# Streamlit web app: upload image -> CNN prediction (MobileNetV2-preprocessed)
# Run: streamlit run app.py

import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
CLASSES_PATH = ARTIFACTS_DIR / "class_names.json"

st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CNN Image Classifier (Streamlit)")
st.write("Upload an image (JPG/PNG), the model will predict its class with confidence.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model tidak ditemukan: artifacts/model.keras")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    if not CLASSES_PATH.exists():
        st.error("class_names.json tidak ditemukan di artifacts/")
        st.stop()
    return json.loads(CLASSES_PATH.read_text())

def preprocess(img: Image.Image, size):
    img = img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

model = load_model()
class_names = load_classes()
input_shape = model.inputs[0].shape
H, W = int(input_shape[1]), int(input_shape[2])
st.caption(f"Input size model: {W}×{H}")

uploaded = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])
top_k = st.slider("Tampilkan Top‑k prediksi", min_value=1, max_value=min(5, len(class_names)), value=3)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Gambar diunggah", use_container_width=True)
    x = preprocess(image, (W, H))
    with st.spinner("Memproses..."):
        preds = model.predict(x, verbose=0)[0]
    order = np.argsort(preds)[::-1][:top_k]

    st.subheader("Hasil Prediksi")
    st.write(f"Top‑1: **{class_names[order[0]]}** ({preds[order[0]]*100:.2f}%)")

    st.subheader("Confidence (Top‑k)")
    st.dataframe(
        {"class": [class_names[i] for i in order],
         "confidence": [float(preds[i]) for i in order]},
        use_container_width=True
    )
    st.bar_chart(np.array([preds[i] for i in order]))

st.markdown("---")
st.caption("Letakkan **model.keras** dan **class_names.json** di folder **artifacts/**. "
           "Latih model dulu dengan `train_transfer.py`.")
