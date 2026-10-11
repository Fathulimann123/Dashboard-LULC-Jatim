import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(page_title="Peta LULC Jawa Timur", layout="wide", page_icon="🌍")

COLOR_MAP = {
    "Bangunan": "#FF0000",   # Merah
    "Pertanian": "#32CD32",  # Hijau Muda
    "Air": "#0000FF",        # Biru
    "Hutan": "#006400",      # Hijau Tua
    "Mangrove": "#800080"    # Ungu
}

st.title("🌍 Peta Interaktif Tutupan Lahan (LULC) Jawa Timur")
st.markdown("Visualisasi spasial hasil pemetaan poligon tutupan lahan dengan fitur filter per kelas.")

# ==============================================================================
# ALUR UNGGAH & BACA DATA
# ==============================================================================
uploaded_file = st.file_uploader("📁 Unggah file GeoJSON (contoh: sampel_uts_jatim_fix.geojson)", type=["geojson"])

if uploaded_file is not None:
    # 1. Membaca GeoJSON
    gdf = gpd.read_file(uploaded_file)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # 2. Membersihkan & Memetakan Nama Kelas Otomatis
    # Jika file asli QGIS yang diupload (hanya punya kolom 'Nama' seperti 'b1', 'p2')
    if "Nama" in gdf.columns:
        peta_awalan = {"b": "Bangunan", "p": "Pertanian", "s": "Pertanian", "a": "Air", "h": "Hutan", "m": "Mangrove"}
        gdf["huruf_depan"] = gdf["Nama"].astype(str).str.strip().str.lower().str[0]
        gdf["Kelas"] = gdf["huruf_depan"].map(peta_awalan)
    elif "kelas" in gdf.columns:
        gdf["Kelas"] = gdf["kelas"].astype(str).str.title()
    elif "Prediksi_LULC" in gdf.columns:
        gdf["Kelas"] = gdf["Prediksi_LULC"]

    # Buang poligon yang tidak memiliki kelas valid
    gdf = gdf.dropna(subset=["Kelas"])

    st.success(f"✓ Berhasil memuat {len(gdf)} poligon area. Membangun peta interaktif...")

    # ==============================================================================
    # RENDER PETA FOLIUM (LAYER CONTROL)
    # ==============================================================================
    centroid = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles=None)

    # Basemap Google Satellite
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Membuat grup centang untuk masing-masing kelas
    grup_layer = {kelas: folium.FeatureGroup(name=kelas, show=True) for kelas in COLOR_MAP.keys()}

    # Memasukkan poligon ke grup warnanya masing-masing
    for _, row in gdf.iterrows():
        kelas_objek = row["Kelas"]
        warna = COLOR_MAP.get(kelas_objek, "#FFFFFF")
        
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, w=warna: {
                'fillColor': w,
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=f"Kelas: {kelas_objek}"
        ).add_to(grup_layer[kelas_objek])

    # Menambahkan semua grup ke peta utama
    for grup in grup_layer.values():
        grup.add_to(m)

    # Mengaktifkan kotak centang (Layer Control)
    folium.LayerControl(collapsed=False).add_to(m)

    # Menampilkan ke Streamlit
    st_folium(m, width=1200, height=600)

else:
    st.info("Silakan unggah file GeoJSON Anda untuk memunculkan peta.")
