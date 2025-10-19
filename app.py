# app.py
# Streamlit app: load model from (1) local artifacts, (2) Secrets URL, or (3) user upload via UI.
# Run: streamlit run app.py

from tensorflow.keras.applications import mobilenet_v2, efficientnet
import json, io, zipfile, re
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import requests

# -------------------- Paths & candidates --------------------
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

st.set_page_config(page_title="CNN Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CNN Image Classifier (Streamlit)")
st.write("Upload an image (JPG/PNG), the model will predict its class with confidence.")

# -------------------- Utils --------------------
def _first_exists(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fix_direct_url(url: str) -> str:
    """Ubah link share Drive/Dropbox menjadi direct download."""
    if "drive.google.com" in url:
        m = re.search(r"/d/([^/]+)/", url) or re.search(r"[?&]id=([^&]+)", url)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in url:
        url = url.replace("?dl=0", "?dl=1").replace("?raw=0", "?raw=1")
    return url

def _download(url: str, dst: Path):
    _ensure_dir(dst.parent)
    url = _fix_direct_url(url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.itercontent(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

def _download_and_extract_zip(url: str, dst_dir: Path) -> Path:
    _ensure_dir(dst_dir)
    url = _fix_direct_url(url)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dst_dir)
    for p in dst_dir.rglob("saved_model.pb"):
        return p.parent
    return dst_dir

def _get_input_size(model, fallback=(224, 224)):
    try:
        ishape = tf.keras.backend.int_shape(model.inputs[0])
        H, W = int(ishape[1]), int(ishape[2])
        return (H or fallback[0], W or fallback[1])
    except Exception:
        return fallback

# -------------------- Loaders --------------------
@st.cache_resource
def load_model_and_classes():
    # 1) Local files?
    model_path = _first_exists(MODEL_CANDIDATES)
    classes_path = _first_exists(CLASS_CANDIDATES)

    # 2) Secrets URL?
    if model_path is None:
        model_url = (st.secrets.get("MODEL_URL", "") or "").strip()
        if model_url:
            if model_url.lower().endswith((".keras", ".h5")):
                target = ROOT / "artifacts" / ("model.keras" if model_url.lower().endswith(".keras") else "model.h5")
                _download(model_url, target)
                model_path = target
            else:
                saved_dir = _download_and_extract_zip(model_url, ROOT / "artifacts" / "saved_model_zip")
                model_path = saved_dir

    if classes_path is None:
        classes_url = (st.secrets.get("CLASSES_URL", "") or "").strip()
        if classes_url:
            target = ROOT / "artifacts" / "class_names.json"
            _download(classes_url, target)
            classes_path = target

    model = tf.keras.models.load_model(str(model_path)) if model_path else None
    classes = None
    if classes_path and classes_path.exists():
        data = json.loads(classes_path.read_text())
        # pastikan list of str
        classes = [str(x) for x in data]
    return model, classes

# tambahkan fungsi untuk deteksi backbone:
def _detect_backbone(m):
    name = (getattr(m, "name", "") or "").lower()
    if "efficientnet" in name: 
        return "efficientnet"
    # fallback: cek nama layer
    for lyr in m.layers:
        if "efficientnet" in lyr.name.lower():
            return "efficientnet"
    return "mobilenet_v2"

BACKBONE = _detect_backbone(model)
st.caption(f"Backbone terdeteksi: {BACKBONE}")

def _save_uploaded_model(file) -> Path:
    """Simpan file model yang di-upload ke artifacts/, dukung .keras/.h5/ZIP(SavedModel)."""
    artifacts = ROOT / "artifacts"
    _ensure_dir(artifacts)
    name = file.name.lower()
    data = file.read()
    if name.endswith((".keras", ".h5")):
        out = artifacts / ("model.keras" if name.endswith(".keras") else "model.h5")
        with open(out, "wb") as f: f.write(data)
        return out
    # assume ZIP SavedModel
    zip_bytes = io.BytesIO(data)
    with zipfile.ZipFile(zip_bytes) as z:
        z.extractall(artifacts / "saved_model_upload")
    return artifacts / "saved_model_upload"

def _save_uploaded_classes(file) -> Path:
    artifacts = ROOT / "artifacts"
    _ensure_dir(artifacts)
    out = artifacts / "class_names.json"
    with open(out, "wb") as f:
        f.write(file.read())
    return out

# --------- Preprocess tanpa distorsi (resize_with_pad) ---------
def preprocess(img: Image.Image, size_wh):
    W, H = size_wh
    img = img.convert("RGB")
    x = np.array(img).astype("float32")
    x = tf.image.resize_with_pad(x, H, W).numpy()
    # pilih preprocessing sesuai backbone
    if BACKBONE == "efficientnet":
        x = efficientnet.preprocess_input(x)
    else:
        x = mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

# --------- Test-Time Augmentation (TTA) untuk inferensi yang lebih stabil ---------
def predict_tta(pil_img, size_wh, rounds=4):
    W, H = size_wh
    cands = [
        pil_img,
        pil_img.transpose(Image.FLIP_LEFT_RIGHT),
        pil_img.resize((W+16, H+16)).resize((W, H)),
        pil_img.resize((W-16, H-16)).resize((W, H)),
    ][:rounds]
    ps = []
    for im in cands:
        x = preprocess(im, (W, H))
        ps.append(model.predict(x, verbose=0)[0])
    return np.mean(ps, axis=0)

# -------------------- Main --------------------
# Try load from local/Secrets
model, class_names = None, None
try:
    model, class_names = load_model_and_classes()
except Exception:
    st.info("Belum ada model/kelas yang valid. Kamu bisa upload di bawah.")

# If not found, allow manual upload via UI
if model is None or class_names is None:
    st.warning("Model/kelas belum tersedia. Upload di bawah atau perbaiki Secrets/letakkan file di artifacts/")
    up_model = st.file_uploader("Upload model (.keras / .h5 / SavedModel.zip)", type=["keras","h5","zip"])
    up_classes = st.file_uploader("Upload class_names.json", type=["json"])
    if up_model and up_classes:
        try:
            model_path = _save_uploaded_model(up_model)
            classes_path = _save_uploaded_classes(up_classes)
            model = tf.keras.models.load_model(str(model_path))
            class_names = [str(x) for x in json.loads(classes_path.read_text())]
            st.success("Model & class_names berhasil dimuat dari upload.")
        except Exception as e:
            st.error(f"Gagal memuat dari upload: {e}")
            st.stop()
    else:
        st.stop()

# Guardrail: pastikan jumlah kelas = unit output
if model.output_shape[-1] != len(class_names):
    st.error(f"Mismatch: output units model = {model.output_shape[-1]} "
             f"≠ jumlah classes = {len(class_names)}. "
             "Upload class_names.json yang sesuai atau retrain model.")
    st.stop()

# Ready to predict
H, W = _get_input_size(model, fallback=(224, 224))
st.caption(f"Input size model: {W}×{H}")

uploaded = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

num_classes = len(class_names)
if num_classes < 2:
    st.info("Model hanya memiliki 1 kelas. Top-k di-set otomatis ke 1.")
    top_k = 1
else:
    max_k = min(5, num_classes)
    default_k = min(3, max_k)
    top_k = st.slider("Tampilkan Top-k prediksi", min_value=1, max_value=max_k, value=default_k)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Gambar diunggah", use_container_width=True)
    with st.spinner("Memproses..."):
        preds = predict_tta(image, (W, H), rounds=4)  # <-- pakai TTA
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
st.caption("Sumber model: artifacts/ atau Secrets (MODEL_URL/CLASSES_URL). Jika keduanya tidak ada, gunakan uploader di atas.")
