import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==============================================================================
st.set_page_config(page_title="Dashboard LULC Jawa Timur", layout="wide", page_icon="🌍")

COLOR_MAP = {
    "Bangunan": "#FF0000",   # Merah
    "Pertanian": "#FBC02D",  # Kuning/Hijau Muda
    "Air": "#2196F3",        # Biru
    "Hutan": "#1B5E20",      # Hijau Tua
    "Mangrove": "#8E24AA"    # Ungu
}

# Data Metrik Hasil Evaluasi
CLASSES = ["Bangunan", "Pertanian", "Air", "Hutan", "Mangrove"]
PRECISION = [92, 49, 93, 69, 84]
RECALL = [97, 46, 84, 79, 71]
F1_SCORE = [95, 47, 88, 74, 77]

# Data Confusion Matrix
CM_DATA = [
    [71, 0, 1, 1, 0],   
    [1, 34, 0, 39, 0],  
    [0, 0, 63, 0, 12],  
    [0, 25, 0, 96, 0],  
    [0, 0, 22, 0, 53]   
]

GEOJSON_PATH = "sampel_uts_jatim_fix.geojson"

# ==============================================================================
# 2. SIDEBAR PROFESIONAL (KIRI)
# ==============================================================================
st.sidebar.markdown("### 🌍 Spasial LULC")
st.sidebar.markdown("**Model:** Random Forest (Tekstur Spasial)")
st.sidebar.markdown("**Sensor:** Sentinel-2A (ESA CDSE)")
st.sidebar.markdown("**Akurasi Final:** 75.84%")
st.sidebar.divider()

st.sidebar.markdown("### 🏷️ 5 KELAS TUTUPAN LAHAN")
for kelas, color in COLOR_MAP.items():
    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; margin-bottom:8px;">
            <span style="height:14px; width:14px; background-color:{color}; border-radius:50%; display:inline-block; margin-right:10px; border: 1px solid #ccc;"></span>
            <span style="font-size: 15px;">{kelas}</span>
        </div>
        """, unsafe_allow_html=True
    )

st.sidebar.divider()
# Identitas di Sidebar
st.sidebar.markdown("**👨‍💻 Dikembangkan Oleh:**")
st.sidebar.markdown("Muhammad Fathul Iman Wahid<br>NIM: 240411100180<br>Proyek Sains Data", unsafe_allow_html=True)

# ==============================================================================
# 3. HEADER & KONTEN METRIK (KANAN/UTAMA)
# ==============================================================================
st.title("📊 Evaluasi Kinerja Model & Pemetaan Spasial")
st.markdown("Visualisasi interaktif hasil prediksi tutupan lahan dan performa metrik algoritma Random Forest.")
st.markdown("<br>", unsafe_allow_html=True)

col_cm, col_bar = st.columns(2)

with col_cm:
    st.markdown("**Confusion Matrix (Data Uji 418 Poligon)**")
    st.caption("Perbandingan kelas aktual lapangan vs hasil prediksi model.")
    
    fig_cm = px.imshow(
        CM_DATA, 
        text_auto=True, 
        color_continuous_scale='Teal',
        x=CLASSES, 
        y=CLASSES,
        labels=dict(x="Hasil Prediksi Model", y="Kelas Ground Truth (Aktual)")
    )
    fig_cm.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_bar:
    st.markdown("**Kinerja Klasifikasi per Kelas (Persentase %)**")
    st.caption("Metrik Precision, Recall, dan F1-Score pada 5 kelas lahan.")
    
    fig_bar = go.Figure(data=[
        go.Bar(name='Precision', y=CLASSES, x=PRECISION, orientation='h', marker_color='#80cbc4'),
        go.Bar(name='Recall', y=CLASSES, x=RECALL, orientation='h', marker_color='#00897b'),
        go.Bar(name='F1-Score', y=CLASSES, x=F1_SCORE, orientation='h', marker_color='#004d40')
    ])
    fig_bar.update_layout(barmode='group', margin=dict(l=20, r=20, t=20, b=20), height=350, yaxis={'categoryorder':'array', 'categoryarray': CLASSES[::-1]})
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ==============================================================================
# 4. PETA INTERAKTIF FOLIUM (LAYER CONTROL & WATERMARK)
# ==============================================================================
st.subheader("🗺️ Peta Distribusi Spasial Tutupan Lahan")
st.markdown("Gunakan **ikon kotak berlapis di pojok kanan atas peta** untuk mencentang/menghilangkan centang (*filtering*) kelas lahan yang ingin dilihat.")

if os.path.exists(GEOJSON_PATH):
    with st.spinner("Memuat data geospasial..."):
        gdf = gpd.read_file(GEOJSON_PATH)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        if "Nama" in gdf.columns:
            peta_awalan = {"b": "Bangunan", "p": "Pertanian", "s": "Pertanian", "a": "Air", "h": "Hutan", "m": "Mangrove"}
            gdf["huruf_depan"] = gdf["Nama"].astype(str).str.strip().str.lower().str[0]
            gdf["Kelas"] = gdf["huruf_depan"].map(peta_awalan)
        elif "kelas" in gdf.columns:
            gdf["Kelas"] = gdf["kelas"].astype(str).str.title()
        
        gdf = gdf.dropna(subset=["Kelas"])

        if not gdf.empty:
            centroid = gdf.geometry.unary_union.centroid
            
            # PETA DIINISIALISASI (Bebas digeser dan di-zoom)
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles=None)

            # Basemap
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='OpenStreetMap',
                overlay=False,
                control=True
            ).add_to(m)

            # Feature Group untuk Filter Native
            grup_layer = {}
            for kelas in COLOR_MAP.keys():
                grup_layer[kelas] = folium.FeatureGroup(name=f"Kelas: {kelas}", show=True)
                m.add_child(grup_layer[kelas])

            for _, row in gdf.iterrows():
                kelas_objek = row["Kelas"]
                warna = COLOR_MAP.get(kelas_objek, "#FFFFFF")
                
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, w=warna: {
                        'fillColor': w,
                        'color': w,           
                        'weight': 1.5,        
                        'fillOpacity': 0.6    
                    },
                    tooltip=folium.Tooltip(f"<b>Kelas Lahan:</b> {kelas_objek}", style="font-size: 14px;")
                ).add_to(grup_layer[kelas_objek])

            # Mengaktifkan Layer Control
            folium.LayerControl(collapsed=False).add_to(m)

            # ==================================================================
            # WATERMARK IDENTITAS MAHASISWA MELAYANG DI PETA
            # ==================================================================
            watermark_html = """
            <div style="
                position: fixed; 
                bottom: 30px; left: 30px; width: 280px; height: auto; 
                background-color: rgba(255, 255, 255, 0.85);
                border: 2px solid #004d40; z-index: 9999; font-size: 13px;
                font-family: Arial, sans-serif; padding: 12px; border-radius: 8px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                <b style="color: #004d40; font-size: 14px;">🎓 Proyek Sains Data</b><br>
                <hr style="margin: 5px 0;">
                <b>Nama:</b> Muhammad Fathul Iman Wahid<br>
                <b>NIM:</b> 240411100180
            </div>
            """
            m.get_root().html.add_child(folium.Element(watermark_html))

            # Render peta ke Streamlit
            st_folium(m, use_container_width=True, height=600, returned_objects=[])
        else:
            st.error("Data valid tidak ditemukan di dalam GeoJSON.")
else:
    st.error(f"File '{GEOJSON_PATH}' tidak ditemukan. Pastikan sudah diunggah ke GitHub.")
