# 🌍 Dashboard Tutupan Lahan Jawa Timur

Peta interaktif untuk menjelajahi hasil klasifikasi **Land Use / Land Cover (LULC)** di Jawa Timur. Proyek ini menampilkan prediksi berbasis citra Sentinel-2A menggunakan Random Forest dengan fitur tekstur spasial, dalam sel berukuran 100 × 100 m.

> Proyek Sains Data · Muhammad Fathul Iman Wahid · 240411100180

## ✨ Yang bisa kamu lihat

- **Peta interaktif** dengan layer prediksi per kelas dan pilihan basemap satelit atau OpenStreetMap.
- **Perkiraan luas tiap kelas** dalam hektare dan persentase area terpetakan.
- **Layer sampel digitasi** sebagai pembanding visual jika berkas GeoJSON tersedia.
- **Evaluasi model** melalui confusion matrix, precision, recall, dan F1-score.

## 🗺️ Kelas dan ringkasan hasil

| Kelas | Luas (ha) | Area terpetakan |
|---|---:|---:|
| Bangunan | 283.327 | 4,50% |
| Pertanian | 3.056.420 | 48,59% |
| Air | 1.537.363 | 24,44% |
| Hutan | 1.365.120 | 21,70% |
| Mangrove | 48.190 | 0,77% |

Evaluasi yang disertakan mencatat akurasi **74,29%** pada **490 sampel validasi lintas wilayah**. Nilai ini merupakan hasil evaluasi yang tersimpan di proyek, bukan jaminan akurasi untuk setiap lokasi. Sel di batas beberapa jenis tutupan lahan tetap mendapat satu label prediksi.

## 🚀 Jalankan di komputer sendiri

Pastikan Python dan pip tersedia. Dari terminal, masuk ke folder `Dashboard-LULC-Jatim`, lalu buat dan aktifkan virtual environment.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Pasang dependensi dan jalankan dashboard:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal dashboard di terminal. Basemap peta memerlukan koneksi internet.

## 📁 Isi proyek

```text
Dashboard-LULC-Jatim/
├── app.py
├── requirements.txt
├── sampel_uts_jatim_fix.geojson
├── model_LULC_jatim_texture.joblib
└── data/
    ├── peta_meta.json
    ├── metrik.json
    ├── luas_per_kelas.csv
    └── kelas_{Bangunan,Pertanian,Air,Hutan,Mangrove}.png
```

- `app.py` adalah aplikasi Streamlit dan peta Folium.
- `data/kelas_*.png` berisi layer raster prediksi per kelas; `peta_meta.json` menyimpan bounds dan warna peta.
- `luas_per_kelas.csv` dan `metrik.json` menyimpan ringkasan luas serta hasil evaluasi.
- `sampel_uts_jatim_fix.geojson` menyediakan layer sampel digitasi. Jika berkas ini tidak tersedia, dashboard tetap dapat berjalan tanpa layer sampel.
- `model_LULC_jatim_texture.joblib` adalah artefak model yang disertakan; dashboard saat ini menampilkan hasil raster dan metrik yang sudah disimpan.

## 🧭 Catatan penggunaan

Data luas dan evaluasi ditampilkan sesuai berkas yang tersedia di proyek. Konsistensi angka tersebut tidak dengan sendirinya memverifikasi batas administratif atau cakupan geografis terhadap sumber acuan eksternal. Gunakan hasil sebagai eksplorasi dan sertakan pemeriksaan data acuan sebelum mengambil keputusan dari peta.
