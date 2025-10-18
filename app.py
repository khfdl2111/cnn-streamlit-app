# app.py
# Streamlit web app: upload image -> CNN prediction (MobileNetV2-preprocessed)
# Run: streamlit run app.py

import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- Paths ---
ROOT = Path(__file__).parent
ARTIFACTS_DIRS = [ROOT / "artifacts", ROOT / "Artifacts"]  # handle case-sensitive mistakes

MODEL_CANDIDATES = []
for d in ARTIFACTS_DIRS:
    MODEL_CANDIDATES += [
        d / "model.keras",
        d / "model.h5",
        d / "saved_model",  # SavedModel dir
    ]

CLASS_CANDIDATES = [(d / "class_names.json") for d in ARTIFACTS_DIRS]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CNN Image Classifier (Streamlit)")
st.write("Upload an image (JPG/PNG), the model will predict its class with confidence.")

def _resolve_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

@st.cache_resource
def load_model():
    model_path = _resolve_first_existing(MODEL_CANDIDATES)
    if model_path is None:
        # List isi folder artifacts jika ada — biar mudah debug
        existing = []
        for d in ARTIFACTS_DIRS:
            if d.exists():
                existing += [str(p) for p in d.glob("**/*")]
        msg = (
            "Model tidak ditemukan. Pastikan salah satu path ini ADA:\n"
            "- artifacts/model.keras\n"
            "- artifacts/model.h5\n"
            "- artifacts/saved_model (folder SavedModel)\n"
            "Catatan: jika file >100MB, unggah via Git LFS."
        )
        st.error(msg)
        if existing:
            with st.expander("Lihat isi folder artifacts saat ini"):
                st.write(existing)
        st.stop()

    # SavedModel (folder) atau file tunggal
    return tf.keras.models.load_model(str(model_path))

@st.cache_data
def load_classes():
    classes_path = _resolve_first_existing(CLASS_CANDIDATES)
    if classes_path is None:
        st.error("`class_names.json` tidak ditemukan di folder artifacts/.")
        st.stop()
    return json.loads(classes_path.read_text())

def preprocess(img: Image.Image, size):
    img = img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

# --- Load assets ---
model = load_model()
class_names = load_classes()

# Derive model input size
ishape = model.inputs[0].shape
H, W = int(ishape[1]), int(ishape[2])
st.caption(f"Input size model: {W}×{H}")

# --- UI ---
uploaded = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
top_k = st.slider("Tampilkan Top-k prediksi", min_value=1, max_value=min(5, len(class_names)), value=3)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Gambar diunggah", use_container_width=True)
    x = preprocess(image, (W, H))
    with st.spinner("Memproses..."):
        preds = model.predict(x, verbose=0)[0]
    order = np.argsort(preds)[::-1][:top_k]

    st.subheader("Hasil Prediksi")
    st.write(f"Top-1: **{class_names[order[0]]}** ({preds[order[0]]*100:.2f}%)")

    st.subheader("Confidence (Top-k)")
    st.dataframe(
        {"class": [class_names[i] for i in order],
         "confidence": [float(preds[i]) for i in order]},
        use_container_width=True
    )
    st.bar_chart(np.array([preds[i] for i in order]))

st.markdown("---")
st.caption(
    "Tempatkan **model.keras**/**model.h5** atau **saved_model/** dan **class_names.json** di folder **artifacts/**. "
    "Jika ukuran model >100MB, gunakan **Git LFS**."
)

