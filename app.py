import os
import tempfile
import joblib
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import openeo
import streamlit as st
from streamlit_folium import st_folium

# ==============================================================================
# 1. KONFIGURASI HALAMAN UTAMA DASHBOARD
# ==============================================================================
st.set_page_config(page_title="Dashboard LULC Jawa Timur", layout="wide", page_icon="🌍")

# Warna standar resmi kelas tutupan lahan
COLOR_MAP = {
    "Bangunan": "#FF0000",   # Merah
    "Pertanian": "#32CD32",  # Hijau Muda
    "Air": "#0000FF",        # Biru
    "Hutan": "#006400",      # Hijau Tua
    "Mangrove": "#800080"    # Ungu
}

# ==============================================================================
# 2. UI: DOKUMENTASI AKADEMIK & BUSINESS UNDERSTANDING
# ==============================================================================
st.title("🌍 Dashboard Klasifikasi Tutupan Lahan (LULC) Jawa Timur")
st.markdown("### Integrasi Kecerdasan Buatan Spasial & Citra Satelit Sentinel-2A")

with st.expander("📖 Business Understanding, Metodologi, & Analisis Spasial (Klik untuk membuka)", expanded=False):
    st.markdown("""
    ### **Business Understanding**
    Pemantauan Tutupan Lahan (LULC) berskala regional sangat krusial untuk mitigasi lingkungan, ketahanan pangan, dan tata ruang. 
    Proyek ini mengadopsi standar metodologi ilmiah dengan **Validasi Spasial Ekstrem (Spatial Cross-Validation)** untuk memastikan ketahanan algoritma.

    ### **Mitigasi Spatial Data Leakage & Optimasi Radius DBSCAN**
    Untuk mencegah *overfitting* akibat efek autokorelasi spasial, data dipisahkan menggunakan *StratifiedGroupKFold* berbasis kluster DBSCAN 10 km.
    
    ### **Performa Model Final (Out-of-Sample Accuracy: 75.84%)**
    Model menggunakan 18 fitur (9 spektral dasar + 9 rekayasa tekstur spasial) hasil kompresi 3,4 juta piksel menjadi 418 poligon.
    - **Bangunan (F1: 0.95)** & **Air (F1: 0.88)**: Sangat presisi berkat indeks NDBI dan NDWI.
    - **Mangrove (F1: 0.77)**: Terkunci baik tanpa bias perairan laut terbuka.
    - **Hutan (F1: 0.74)** & **Pertanian (F1: 0.47)**: Performa membaik tajam berkat penambahan sampel & fitur tekstur spasial (`_std`) untuk membedakan tajuk pohon dan kanopi sawah.
    """)

# ==============================================================================
# 3. BACKEND: PROSES INFERENSI PYTHON & CRAWLING CITRA via openEO
# ==============================================================================
@st.cache_resource
def load_model():
    return joblib.load("model_LULC_jatim_texture.joblib")

def process_and_predict(gdf_input, model_data):
    st.info("🔄 Menghubungkan ke Copernicus Data Space Ecosystem (CDSE)...")
    
    host_cdse = "openeo.dataspace.copernicus.eu"
    conn = openeo.connect(host_cdse)
    
    # Otentikasi aman via Streamlit Secrets (atau fallback ke OIDC standar)
    try:
        conn.authenticate_oidc(
            client_id=st.secrets.get("CDSE_CLIENT_ID", ""),
            client_secret=st.secrets.get("CDSE_CLIENT_SECRET", "")
        )
    except Exception:
        conn.authenticate_oidc()
    
    epsilon = 1e-9
    fitur_utama = ["b02", "b03", "b04", "b08", "b11", "ndvi", "ndwi", "mndwi", "ndbi"]
    hasil_prediksi = []
    
    progress_bar = st.progress(0)
    total_polygons = len(gdf_input)
    
    # Ekstraksi Bounding Box
    minx, miny, maxx, maxy = gdf_input.total_bounds
    bbox = {
        "west": float(minx - 0.005), "south": float(miny - 0.005),
        "east": float(maxx + 0.005), "north": float(maxy + 0.005)
    }
    
    st.info("📡 Mengunduh citra Sentinel-2 komposit median (September 2026)...")
    cube = conn.load_collection(
        "SENTINEL2_L2A", spatial_extent=bbox,
        temporal_extent=["2026-09-01", "2026-09-30"],
        bands=["B02", "B03", "B04", "B08", "B11"]
    )
    
    # Download file TIF sementara
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
        nama_tif = tmp_tif.name
    
    cube.median_time().download(nama_tif, format="GTiff")
    
    st.info("⚙️ Mengekstraksi 18 Fitur Spektral & Tekstur Spasial...")
    with rasterio.open(nama_tif) as src:
        gdf_proj = gdf_input.to_crs(src.crs)
        
        for idx, row in gdf_proj.iterrows():
            try:
                arr, _ = mask(src, [row.geometry], crop=True, nodata=0, filled=True)
            except ValueError:
                hasil_prediksi.append(None)
                progress_bar.progress((idx + 1) / total_polygons)
                continue
                
            b2, b3, b4, b8, b11 = arr[0], arr[1], arr[2], arr[3], arr[4]
            
            # Perhitungan Indeks + Epsilon
            ndvi = (b8 - b4) / (b8 + b4 + epsilon)
            ndwi = (b3 - b8) / (b3 + b8 + epsilon)
            mndwi = (b3 - b11) / (b3 + b11 + epsilon)
            ndbi = (b11 - b8) / (b11 + b8 + epsilon)
            
            valid_mask = (b2 > 0) & (b3 > 0) & (b4 > 0)
            
            pixels = {
                "b02": b2[valid_mask], "b03": b3[valid_mask], 
                "b04": b4[valid_mask], "b08": b8[valid_mask], "b11": b11[valid_mask],
                "ndvi": ndvi[valid_mask], "ndwi": ndwi[valid_mask], 
                "mndwi": mndwi[valid_mask], "ndbi": ndbi[valid_mask]
            }
            
            # Hitung Rata-rata & Standar Deviasi
            fitur_siap = {}
            for f in fitur_utama:
                fitur_siap[f"{f}_mean"] = np.mean(pixels[f]) if len(pixels[f]) > 0 else 0
                fitur_siap[f"{f}_std"] = np.std(pixels[f]) if len(pixels[f]) > 0 else 0
            
            # Menyelaraskan urutan fitur agar 100% sama dengan saat model dilatih
            X_input = [fitur_siap[f] for f in model_data['fitur']]
            
            # Eksekusi Prediksi
            pred_idx = model_data['model'].predict([X_input])[0]
            nama_kelas_mentah = model_data['target_names'][pred_idx]
            nama_kelas = nama_kelas_mentah.split(" ")[0] # Ambil kata utama saja
            hasil_prediksi.append(nama_kelas)
            
            progress_bar.progress((idx + 1) / total_polygons)
            
    # Bersihkan memori disk
    if os.path.exists(nama_tif):
        os.remove(nama_tif)
    
    gdf_input["Prediksi_LULC"] = hasil_prediksi
    return gdf_input.dropna(subset=["Prediksi_LULC"])

# ==============================================================================
# 4. TAMPILAN VISUALISASI PETA INTERAKTIF LAYER CONTROL (FOLIUM UTAMA)
# ==============================================================================
def render_map(gdf_predicted):
    centroid = gdf_predicted.to_crs("EPSG:4326").geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles=None)
    
    # Basemap Google Satellite
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # FeatureGroup dinamis untuk tiap kelas (Bisa di-centang / un-centang user)
    grup_layer = {kelas: folium.FeatureGroup(name=f"{kelas}", show=True) for kelas in COLOR_MAP.keys()}
    
    for _, row in gdf_predicted.iterrows():
        kelas_prediksi = row["Prediksi_LULC"]
        warna = COLOR_MAP.get(kelas_prediksi, "#FFFFFF")
        
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, warna_isi=warna: {
                'fillColor': warna_isi,
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=f"Klasifikasi AI: {kelas_prediksi}"
        ).add_to(grup_layer[kelas_prediksi])
    
    for grup in grup_layer.values():
        grup.add_to(m)
        
    # Memunculkan kotak Layer Control interaktif
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m

# ==============================================================================
# 5. ALUR UTAMA PENGGUNA (USER INTERACTION)
# ==============================================================================
st.subheader("📁 Unggah Dokumen Area (Sampel Uji Baru)")
uploaded_file = st.file_uploader("Pilih file batas area berformat GeoJSON (.geojson)", type=["geojson"])

try:
    model_data = load_model()
    st.success("🤖 Berkas AI 'model_LULC_jatim_texture.joblib' sukses dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model! Pastikan file model ada di folder yang sama. Info: {e}")
    st.stop()

if uploaded_file is not None:
    # Membaca file GeoJSON yang diunggah
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp_json:
        tmp_json.write(uploaded_file.getvalue())
        nama_json = tmp_json.name
        
    gdf_input = gpd.read_file(nama_json)
    if gdf_input.crs != "EPSG:4326":
        gdf_input = gdf_input.to_crs("EPSG:4326")
        
    os.remove(nama_json)
    st.write(f"Total data terbaca: **{len(gdf_input)} poligon area**.")
    
    if st.button("🚀 Mulai Proses Prediksi & Klasifikasi"):
        with st.spinner("Model sedang mendownload & memproses citra satelit... Mohon tunggu."):
            try:
                gdf_hasil = process_and_predict(gdf_input, model_data)
                st.success("✓ Selesai! Klasifikasi berhasil dilakukan.")
                
                # Visualisasi Peta
                st.markdown("### 🗺️ Peta Hasil Klasifikasi LULC Interaktif")
                st.markdown("*Gunakan menu centang di sudut kanan atas peta untuk menyaring tampilan per kelas (misal: Sawah/Pertanian saja).*")
                
                peta_final = render_map(gdf_hasil)
                st_folium(peta_final, width=1200, height=600)
                
                # Ringkasan & Download
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Statistik Area")
                    st.dataframe(gdf_hasil["Prediksi_LULC"].value_counts().rename_axis("Kelas").reset_index(name="Jumlah Poligon"))
                with col2:
                    st.write("") 
                    st.write("")
                    st.download_button(
                        label="📥 Unduh Hasil (.geojson)",
                        data=gdf_hasil.to_json(),
                        file_name="prediksi_lulc_jatim.geojson",
                        mime="application/geo+json"
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis saat eksekusi: {e}")