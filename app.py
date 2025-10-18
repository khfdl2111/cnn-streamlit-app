# app.py
# Streamlit app with local-or-remote model loading (robust version)
# Run: streamlit run app.py

import json, io, zipfile, re
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import requests

# ------------------------- Paths & Candidates -------------------------
ROOT = Path(__file__).parent
ART_DIRS = [ROOT / "artifacts", ROOT / "Artifacts"]  # antisipasi typo kapital

MODEL_CANDIDATES = []
for d in ART_DIRS:
    MODEL_CANDIDATES += [
        d / "model.keras",
        d / "model.h5",
        d / "model_last.keras",
        d / "model_last.h5",
        d / "saved_model",  # folder SavedModel
    ]
CLASS_CANDIDATES = [(d / "class_names.json") for d in ART_DIRS]

# ------------------------- UI Header -------------------------
st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CNN Image Classifier (Streamlit)")
st.write("Upload an image (JPG/PNG), the model will predict its class with confidence.")

# ------------------------- Helpers -------------------------
def _first_exists(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fix_direct_url(url: str) -> str:
    """Konversi link share menjadi direct download untuk Drive/Dropbox."""
    if "drive.google.com" in url:
        # format 1: https://drive.google.com/file/d/<ID>/view?usp=sharing
        m = re.search(r"/d/([^/]+)/", url)
        if m:
            file_id = m.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        # format 2: ...?id=<ID>
        m = re.search(r"[?&]id=([^&]+)", url)
        if m:
            file_id = m.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    if "dropbox.com" in url:
        # ubah ?dl=0 -> ?dl=1
        if "?dl=0" in url:
            return url.replace("?dl=0", "?dl=1")
        if "?raw=0" in url:
            return url.replace("?raw=0", "?raw=1")
    return url

def _download(url: str, dst: Path):
    _ensure_dir(dst.parent)
    url = _fix_direct_url(url)
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _download_and_extract_zip(url: str, dst_dir: Path) -> Path:
    """Unduh ZIP dan ekstrak ke dst_dir. Kembalikan folder SavedModel yang valid."""
    _ensure_dir(dst_dir)
    url = _fix_direct_url(url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dst_dir)
    # cari folder yang berisi saved_model.pb
    for p in dst_dir.rglob("saved_model.pb"):
        return p.parent
    return dst_dir  # fallback (jika ZIP memang sudah berisi folder saved_model di root)

def _get_input_size(model, fallback=(224, 224)):
    try:
        ishape = tf.keras.backend.int_shape(model.inputs[0])
        H, W = int(ishape[1]), int(ishape[2])
        if H is None or W is None:
            return fallback
        return (H, W)
    except Exception:
        return fallback

# ------------------------- Loaders -------------------------
@st.cache_resource
def load_model():
    # 1) coba lokal
    local_model = _first_exists(MODEL_CANDIDATES)

    # 2) coba unduh dari secrets jika belum ada
    if local_model is None:
        model_url = (st.secrets.get("MODEL_URL", "") or "").strip()
        if model_url:
            # tentukan target: file .keras/.h5 atau ZIP SavedModel
            if model_url.lower().endswith((".keras", ".h5")):
                target = ROOT / "artifacts" / ("model.keras" if model_url.lower().endswith(".keras") else "model.h5")
                try:
                    _download(model_url, target)
                    local_model = target
                except Exception as e:
                    st.error(f"Gagal mengunduh model: {e}")
                    st.stop()
            else:
                # anggap ZIP SavedModel
                try:
                    target_dir = ROOT / "artifacts" / "saved_model_zip"
                    saved_dir = _download_and_extract_zip(model_url, target_dir)
                    local_model = saved_dir
                except Exception as e:
                    st.error(f"Gagal mengunduh/ekstrak SavedModel ZIP: {e}")
                    st.stop()
        else:
            st.error(
                "Model tidak ditemukan.\n"
                "Letakkan salah satu path berikut di repo **atau** isi `MODEL_URL` di Secrets (direct download):\n"
                "- artifacts/model.keras\n- artifacts/model.h5\n- artifacts/saved_model/ (folder)\n"
                "Catatan: untuk Google Drive/Dropbox, gunakan link **direct**."
            )
            st.stop()

    try:
        return tf.keras.models.load_model(str(local_model))
    except Exception as e:
        st.error(f"Gagal load model dari '{local_model}': {e}")
        st.stop()

@st.cache_data
def load_classes():
    classes_path = _first_exists(CLASS_CANDIDATES)
    if classes_path is None:
        classes_url = (st.secrets.get("CLASSES_URL", "") or "").strip()
        if classes_url:
            target = ROOT / "artifacts" / "class_names.json"
            try:
                _download(classes_url, target)
                classes_path = target
            except Exception as e:
                st.error(f"Gagal mengunduh class_names.json: {e}")
                st.stop()
        else:
            st.error("`class_names.json` tidak ditemukan di artifacts/. Atau isi `CLASSES_URL` di Secrets (direct link).")
            st.stop()
    try:
        data = json.loads(classes_path.read_text())
        if not isinstance(data, list) or not all(isinstance(x, (str, int)) for x in data):
            raise ValueError("Isi class_names.json harus berupa list nama kelas.")
        return [str(x) for x in data]
    except Exception as e:
        st.error(f"Format class_names.json tidak valid: {e}")
        st.stop()

def preprocess(img: Image.Image, size):
    img = img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

# ------------------------- App Logic -------------------------
model = load_model()
class_names = load_classes()

H, W = _get_input_size(model, fallback=(224, 224))
st.caption(f"Input size model: {W}×{H}")

uploaded = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# slider yang aman (default tidak melebihi jumlah kelas)
max_k = max(1, min(5, len(class_names)))
default_k = min(3, max_k)
top_k = st.slider("Tampilkan Top-k prediksi", min_value=1, max_value=max_k, value=default_k)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Gambar diunggah", use_container_width=True)
    x = preprocess(image, (W, H))
    with st.spinner("Memproses..."):
        preds = model.predict(x, verbose=0)[0]  # (num_classes,)
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
    "Taruh model di **artifacts/** atau set `MODEL_URL`/`CLASSES_URL` (direct link). "
    "Jika model >100MB di GitHub, gunakan **Git LFS**. "
    "Pastikan `requirements.txt` memuat `requests` (wajib)."
)
