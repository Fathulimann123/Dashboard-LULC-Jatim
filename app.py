import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Peta LULC Jawa Timur", layout="wide", page_icon="🌍")

COLOR_MAP = {
    "Bangunan": "#FF0000",   
    "Pertanian": "#32CD32",  
    "Air": "#0000FF",        
    "Hutan": "#006400",      
    "Mangrove": "#800080"    
}

st.title("🌍 Peta Interaktif Tutupan Lahan (LULC) Jawa Timur")
st.markdown("Visualisasi spasial hasil pemetaan poligon tutupan lahan dengan fitur filter per kelas.")

# Baca langsung dari file yang ada di GitHub
file_path = "sampel_uts_jatim_fix.geojson"

if os.path.exists(file_path):
    with st.spinner("Memuat peta..."):
        gdf = gpd.read_file(file_path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        if "Nama" in gdf.columns:
            peta_awalan = {"b": "Bangunan", "p": "Pertanian", "s": "Pertanian", "a": "Air", "h": "Hutan", "m": "Mangrove"}
            gdf["huruf_depan"] = gdf["Nama"].astype(str).str.strip().str.lower().str[0]
            gdf["Kelas"] = gdf["huruf_depan"].map(peta_awalan)
        elif "kelas" in gdf.columns:
            gdf["Kelas"] = gdf["kelas"].astype(str).str.title()
        elif "Prediksi_LULC" in gdf.columns:
            gdf["Kelas"] = gdf["Prediksi_LULC"]

        gdf = gdf.dropna(subset=["Kelas"])

        centroid = gdf.geometry.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles=None)

        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        grup_layer = {kelas: folium.FeatureGroup(name=kelas, show=True) for kelas in COLOR_MAP.keys()}

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

        for grup in grup_layer.values():
            grup.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        st_folium(m, width=1200, height=600)
else:
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file tersebut sudah di-upload ke GitHub.")
