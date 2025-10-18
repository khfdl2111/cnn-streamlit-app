# app.py
# Streamlit app with local-or-remote model loading
# Run: streamlit run app.py

import json, io, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import requests

ROOT = Path(__file__).parent
ARTIFACTS_DIRS = [ROOT / "artifacts", ROOT / "Artifacts"]  # handle typo kapital
MODEL_CANDIDATES = []
for d in ARTIFACTS_DIRS:
    MODEL_CANDIDATES += [d / "model.keras", d / "model.h5", d / "saved_model"]
CLASS_CANDIDATES = [(d / "class_names.json") for d in ARTIFACTS_DIRS]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CNN Image Classifier (Streamlit)")
st.write("Upload an image (JPG/PNG), the model will predict its class with confidence.")

def _first_exists(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _download(url: str, dst: Path):
    _ensure_dir(dst.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def _download_and_extract_zip(url: str, dst_dir: Path):
    _ensure_dir(dst_dir)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dst_dir)

@st.cache_resource
def load_model():
    # 1) coba lokal
    local_model = _first_exists(MODEL_CANDIDATES)
    if local_model is None:
        # 2) coba unduh dari secrets
        model_url = st.secrets.get("MODEL_URL", "").strip()
        if model_url:
            # tentukan target: file .keras/.h5 atau folder saved_model
            if model_url.endswith((".keras", ".h5")):
                target = ROOT / "artifacts" / ("model.keras" if model_url.endswith(".keras") else "model.h5")
                try:
                    _download(model_url, target)
                    local_model = target
                except Exception as e:
                    st.error(f"Gagal mengunduh model: {e}")
                    st.stop()
            else:
                # anggap ZIP SavedModel
                target_dir = ROOT / "artifacts" / "saved_model"
                try:
                    _download_and_extract_zip(model_url, target_dir)
                    local_model = target_dir
                except Exception as e:
                    st.error(f"Gagal mengunduh dan ekstrak SavedModel: {e}")
                    st.stop()
        else:
            st.error(
                "Model tidak ditemukan.\n"
                "Letakkan salah satu dari ini di repo **atau** isi `MODEL_URL` di Secrets:\n"
                "- artifacts/model.keras\n- artifacts/model.h5\n- artifacts/saved_model (folder SavedModel/ZIP URL)"
            )
            st.stop()

    return tf.keras.models.load_model(str(local_model))

@st.cache_data
def load_classes():
    classes_path = _first_exists(CLASS_CANDIDATES)
    if classes_path is None:
        classes_url = st.secrets.get("CLASSES_URL", "").strip()
        if classes_url:
            target = ROOT / "artifacts" / "class_names.json"
            try:
                _download(classes_url, target)
                classes_path = target
            except Exception as e:
                st.error(f"Gagal mengunduh class_names.json: {e}")
                st.stop()
        else:
            st.error("`class_names.json` tidak ditemukan di artifacts/. Atau isi `CLASSES_URL` di Secrets.")
            st.stop()
    return json.loads(classes_path.read_text())

def preprocess(img: Image.Image, size):
    img = img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

# Load artifacts
model = load_model()
class_names = load_classes()
H, W = int(model.inputs[0].shape[1]), int(model.inputs[0].shape[2])
st.caption(f"Input size model: {W}×{H}")

uploaded = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
top_k = st.slider("Tampilkan Top-k prediksi", 1, min(5, len(class_names)), 3)

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
st.caption("Letakkan model di **artifacts/** atau isi `MODEL_URL` dan `CLASSES_URL` di Secrets. "
           "Jika ukuran model >100MB dan ingin commit ke repo, gunakan Git LFS.")
