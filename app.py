# Revised version of the script with improved structure and fixes for download behavior

from model_loader import ModelLoader
from image_processor import ImageProcessor
from pdf_generator import PDFGenerator
from PIL import Image
import streamlit as st
import numpy as np

# Judul aplikasi
st.title("Perbandingan Segmentasi Retina U-Net: CLAHE vs Non-CLAHE")

# Inisialisasi session state
if "results" not in st.session_state:
    st.session_state.results = []

if "clahe_inputs" not in st.session_state:
    st.session_state.clahe_inputs = []

if "clahe_results" not in st.session_state:
    st.session_state.clahe_results = []

# Sidebar: Pilihan model segmentasi
model_choice = st.sidebar.selectbox(
    "Pilih Model Segmentasi yang Digunakan",
    ("Keduanya (CLAHE & Non-CLAHE)", "Hanya CLAHE", "Hanya Non-CLAHE")
)

# Sidebar: Opsi hanya CLAHE (tanpa segmentasi)
apply_clahe_only = st.sidebar.checkbox("Terapkan CLAHE (tanpa segmentasi)?")

# Slider konfigurasi CLAHE
if apply_clahe_only:
    clip_limit = st.sidebar.slider("Clip Limit", 1.0, 20.0, 4.0, step=0.5)
    grid_rows = st.sidebar.slider("Grid Rows", 2, 16, 4)
    grid_cols = st.sidebar.slider("Grid Columns", 2, 16, 4)

# Load model & processor
loader = ModelLoader()
model_clahe, model_no_clahe = loader.load_models()
processor = ImageProcessor()

model_required = {
    "Keduanya (CLAHE & Non-CLAHE)": (model_clahe, model_no_clahe),
    "Hanya CLAHE": (model_clahe, None),
    "Hanya Non-CLAHE": (None, model_no_clahe),
}
model_c, model_n = model_required[model_choice]

# Upload gambar retina
uploaded_files = st.file_uploader(
    "Upload beberapa gambar retina (grayscale)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# Mode: Hanya penerapan CLAHE
if uploaded_files and apply_clahe_only:
    st.session_state.clahe_inputs = []
    st.markdown("### Hasil Penerapan CLAHE dan Histogram:")
    for file in uploaded_files:
        try:
            image = Image.open(file).convert("L").resize((128, 128))
            image_np = np.array(image)

            clahe_img = processor.HistogramEqualizationClaheGrayscale(
                image_np, clip_limit=clip_limit, grid_size=(grid_rows, grid_cols)
            )

            st.session_state.clahe_inputs.append({
                "filename": file.name,
                "original": image_np,
                "clahe": clahe_img
            })

            st.markdown(f"#### {file.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_np, caption="Gambar Asli", use_container_width=True)
                st.bar_chart(processor.CalHistogram(image_np))
            with col2:
                st.image(clahe_img, caption="Setelah CLAHE", use_container_width=True)
                st.bar_chart(processor.CalHistogram(clahe_img))
            st.markdown("---")

        except Exception as e:
            st.error(f"Gagal memproses {file.name}: {e}")

# Proses segmentasi mode CLAHE-only
if apply_clahe_only and st.session_state.clahe_inputs:
    if st.button("🔍 Proses Segmentasi"):
        st.session_state.clahe_results = []
        st.markdown("### Hasil Segmentasi Setelah CLAHE:")
        for item in st.session_state.clahe_inputs:
            filename = item["filename"]
            clahe_np = item["clahe"]
            entry = {"filename": filename}
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(clahe_np, caption="Input: Setelah CLAHE", use_container_width=True)
                entry["original"] = Image.fromarray(clahe_np).resize((128, 128))
            if model_clahe:
                input_c, _ = processor.preprocess(Image.fromarray(clahe_np), use_clahe=False)
                pred_c = processor.predict(input_c, model_clahe).resize((128, 128))
                with col2:
                    st.image(pred_c, caption="Segmentasi: U-Net + CLAHE", use_container_width=True)
                entry["clahe"] = pred_c
            if model_no_clahe:
                input_n, _ = processor.preprocess(Image.fromarray(clahe_np), use_clahe=False)
                pred_n = processor.predict(input_n, model_no_clahe).resize((128, 128))
                with col3:
                    st.image(pred_n, caption="Segmentasi: U-Net Tanpa CLAHE", use_container_width=True)
                entry["non_clahe"] = pred_n
            st.session_state.clahe_results.append(entry)
            st.markdown("---")

# Tombol PDF hanya muncul jika hasil CLAHE-only tersedia
if apply_clahe_only and st.session_state.clahe_results:
    export_results = []
    for item in st.session_state.clahe_results:
        ori = item.get("original")
        clahe = item.get("clahe", ori)
        non_clahe = item.get("non_clahe", ori)
        filename = item.get("filename", "unknown")
        export_results.append((ori, clahe, non_clahe, filename))
    try:
        pdf_buffer = PDFGenerator().generate(export_results)
        st.download_button(
            label="📥 Unduh PDF (Segmentasi CLAHE)",
            data=pdf_buffer,
            file_name="segmentasi_clahe_only.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Gagal membuat PDF: {e}")

# Proses segmentasi dari model_choice utama
if uploaded_files and not apply_clahe_only:
    if st.button("🔍 Proses Segmentasi"):
        if (model_choice == "Keduanya (CLAHE & Non-CLAHE)" and (model_c is None or model_n is None)) or \
           (model_choice == "Hanya CLAHE" and model_c is None) or \
           (model_choice == "Hanya Non-CLAHE" and model_n is None):
            st.error("❗ Model belum tersedia.")
        else:
            st.session_state.results = []
            for file in uploaded_files:
                try:
                    image = Image.open(file)
                    image_display = image.resize((128, 128))
                    result_entry = {"filename": file.name, "original": image_display}
                    if model_c:
                        input_c, _ = processor.preprocess(image, use_clahe=True)
                        result_entry["clahe"] = processor.predict(input_c, model_c).resize((128, 128))
                    if model_n:
                        input_n, _ = processor.preprocess(image, use_clahe=False)
                        result_entry["non_clahe"] = processor.predict(input_n, model_n).resize((128, 128))
                    st.session_state.results.append(result_entry)
                except Exception as e:
                    st.error(f"Gagal memproses {file.name}: {e}")

# Tampilkan hasil segmentasi mode utama
if st.session_state.results and not apply_clahe_only:
    st.markdown("### Hasil Segmentasi:")
    for res in st.session_state.results:
        st.markdown(f"#### {res['filename']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(res["original"], caption="Gambar Asli", use_container_width=True)
        with col2:
            if "clahe" in res:
                st.image(res["clahe"], caption="U-Net + CLAHE", use_container_width=True)
        with col3:
            if "non_clahe" in res:
                st.image(res["non_clahe"], caption="U-Net Tanpa CLAHE", use_container_width=True)
        st.markdown("---")

    # Tombol PDF mode utama
    export_results = []
    for item in st.session_state.results:
        ori = item.get("original")
        clahe = item.get("clahe", ori)
        non_clahe = item.get("non_clahe", ori)
        filename = item.get("filename", "unknown")
        export_results.append((ori, clahe, non_clahe, filename))
    try:
        pdf_buffer = PDFGenerator().generate(export_results)
        st.download_button(
            label="📥 Unduh PDF (Perbandingan Segmentasi)",
            data=pdf_buffer,
            file_name="perbandingan_segmentasi.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Gagal membuat PDF: {e}")
