import streamlit as st
import tensorflow as tf
import tempfile

class ModelLoader:
    def __init__(self):
        self.model_clahe = None
        self.model_no_clahe = None

    def load_models(self):
        st.sidebar.title("Model Loader")

        uploaded_model_clahe = st.sidebar.file_uploader("Upload Model U-Net + CLAHE (.h5)", type=["h5"], key="clahe")
        uploaded_model_no_clahe = st.sidebar.file_uploader("Upload Model U-Net Tanpa CLAHE (.h5)", type=["h5"], key="no_clahe")

        # Cek validitas sebelum memuat model
        if uploaded_model_clahe is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
                    temp_file.write(uploaded_model_clahe.read())
                    temp_file_path = temp_file.name
                self.model_clahe = tf.keras.models.load_model(temp_file_path, compile=False)
                st.sidebar.success("Model CLAHE berhasil dimuat.")
            except Exception as e:
                st.sidebar.error(f"Gagal memuat model CLAHE: {e}")
                self.model_clahe = None

        if uploaded_model_no_clahe is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
                    temp_file.write(uploaded_model_no_clahe.read())
                    temp_file_path = temp_file.name
                self.model_no_clahe = tf.keras.models.load_model(temp_file_path, compile=False)
                st.sidebar.success("Model Non-CLAHE berhasil dimuat.")
            except Exception as e:
                st.sidebar.error(f"Gagal memuat model Non-CLAHE: {e}")
                self.model_no_clahe = None

        return self.model_clahe, self.model_no_clahe
