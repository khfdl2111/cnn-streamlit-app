import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# === Load model (pastikan file model_cnn.h5 ada di repo) ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_cnn.h5")
    return model

model = load_model()

# === Judul Aplikasi ===
st.title("🧠 Aplikasi Klasifikasi Gambar dengan CNN")
st.write("Unggah gambar dan lihat hasil prediksi model Convolutional Neural Network (CNN).")

# === Upload Gambar ===
uploaded_file = st.file_uploader("Pilih gambar untuk diklasifikasi:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("⏳ Sedang memproses...")

    # Preprocessing gambar
    img = image.resize((64, 64))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Hasil prediksi
    st.success(f"🎯 Hasil Prediksi: **Kelas {pred_class}** dengan tingkat keyakinan {confidence:.2f}%")
else:
    st.info("Silakan unggah file gambar (.jpg, .jpeg, .png) untuk mulai prediksi.")
