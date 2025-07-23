# 🧠 Segmentasi Retina OCT dengan U-Net dan CLAHE

Proyek ini adalah aplikasi berbasis Python menggunakan **Streamlit** untuk mendeteksi kelainan pada citra **Optical Coherence Tomography (OCT)** retina. Sistem ini menggabungkan model **U-Net** untuk segmentasi dan **CLAHE** (Contrast Limited Adaptive Histogram Equalization) sebagai preprocessing untuk peningkatan kontras citra.

## 🔬 Latar Belakang

Deteksi dini penyakit retina seperti **Age-related Macular Degeneration (AMD)** sangat penting untuk mencegah kebutaan. OCT memungkinkan visualisasi struktur retina secara detail, namun proses interpretasi manual memerlukan keahlian khusus. Aplikasi ini mengotomatiskan proses segmentasi citra OCT untuk membantu diagnosa.

## 🚀 Fitur

- Segmentasi otomatis menggunakan model **U-Net**
- Opsi preprocessing **CLAHE**
- Perbandingan hasil segmentasi dengan dan tanpa CLAHE
- Visualisasi histogram citra asli dan hasil CLAHE
- Ekspor hasil segmentasi dalam bentuk file **PDF**

## 🗂️ Struktur Folder

segmentasi-retina/
- ├── app.py # Aplikasi utama Streamlit
- ├── model_loader.py # Loader model U-Net
- ├── image_processor.py # Modul preprocessing & prediksi
- ├── pdf_generator.py # Modul export hasil ke PDF
- ├── models/ # Folder berisi model CLAHE & non-CLAHE



## ⚙️ Instalasi

1. **Clone repository:**
```bash
git clone https://github.com/username/segmentasi-retina.git
cd segmentasi-retina
```

2. **Buat virtual environment(Opsional)**
```bash
python -m venv venv
source venv/bin/activate  # di Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Jalankan Aplikasi Streamlit**
```bash
streamlit run app.py
```

## 💡 Cara Penggunaan
1. Pilih model segmentasi: CLAHE, Non-CLAHE, atau keduanya
2. Upload Model yang tersedia
3. Upload beberapa citra retina dalam format .jpg, .png, atau .jpeg
4. Jika ingin menerapkan hanya CLAHE, centang "Terapkan CLAHE (tanpa segmentasi)"
5. Klik tombol Proses Segmentasi untuk melihat hasil
6. Unduh hasil sebagai PDF jika diinginkan
