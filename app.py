import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_cnn.h5")

model = load_model()

st.title("🧠 CIFAR-10 Image Classifier (Streamlit)")
st.write("Upload gambar (objek umum) ukuran bebas; app akan resize ke 32×32.")

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    # Preprocess ke 32x32x3
    img_resized = img.resize((32,32))
    x = np.array(img_resized).astype("float32")/255.0
    x = np.expand_dims(x, axis=0)

    # Prediksi
    prob = model.predict(x)[0]       # shape (10,)
    idx = int(np.argmax(prob))
    conf = float(np.max(prob))*100.0
    st.success(f"Prediksi: **{CLASS_NAMES[idx]}** ({conf:.2f}%)")

    # (opsional) tampilkan top-3
    top3_idx = np.argsort(prob)[-3:][::-1]
    st.write("Top-3:")
    for i in top3_idx:
        st.write(f"- {CLASS_NAMES[i]}: {prob[i]*100:.2f}%")
else:
    st.info("Silakan unggah gambar untuk klasifikasi.")
