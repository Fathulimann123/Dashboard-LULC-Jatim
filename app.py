import json
import os

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

# ==============================================================================
# 1. KONFIGURASI HALAMAN & BERKAS DATA
# ==============================================================================
st.set_page_config(page_title="Dashboard LULC Jawa Timur", layout="wide", page_icon="🌍")

DATA_DIR = "data"
CLASSES = ["Bangunan", "Pertanian", "Air", "Hutan", "Mangrove"]

# Cek otomatis apakah geojson ada di root atau di dalam folder data/
if os.path.exists("sampel_uts_jatim_fix.geojson"):
    GEOJSON_PATH = "sampel_uts_jatim_fix.geojson"
elif os.path.exists(f"{DATA_DIR}/sampel_uts_jatim_fix.geojson"):
    GEOJSON_PATH = f"{DATA_DIR}/sampel_uts_jatim_fix.geojson"
else:
    GEOJSON_PATH = None

# Pastikan berkas hasil notebook ada sebelum aplikasi lanjut
BERKAS_WAJIB = ["peta_meta.json", "metrik.json", "luas_per_kelas.csv"] + [f"kelas_{k}.png" for k in CLASSES]
hilang = [b for b in BERKAS_WAJIB if not os.path.exists(f"{DATA_DIR}/{b}")]
if hilang:
    st.error(f"Berkas berikut tidak ditemukan di folder '{DATA_DIR}/': {', '.join(hilang)}")
    st.stop()


# ==============================================================================
# 2. FUNGSI PEMUAT DATA (Diberi parameter mtime agar cache otomatis update!)
# ==============================================================================
@st.cache_data
def muat_meta(_mtime):
    with open(f"{DATA_DIR}/peta_meta.json") as f:
        meta = json.load(f)
    warna = {k: "#%02x%02x%02x" % tuple(v) for k, v in meta["warna"].items()}
    return meta["bounds"], warna


@st.cache_data
def muat_metrik(_mtime):
    with open(f"{DATA_DIR}/metrik.json") as f:
        return json.load(f)


@st.cache_data
def muat_luas(_mtime):
    return pd.read_csv(f"{DATA_DIR}/luas_per_kelas.csv")


@st.cache_data
def muat_sampel(path_geojson, _mtime):
    gdf = gpd.read_file(path_geojson).to_crs("EPSG:4326")
    peta_awalan = {"b": "Bangunan", "p": "Pertanian", "s": "Pertanian",
                   "a": "Air", "h": "Hutan", "m": "Mangrove"}
    huruf = gdf["Nama"].astype(str).str.strip().str.lower().str[0]
    gdf["Kelas"] = huruf.map(peta_awalan)
    return gdf.dropna(subset=["Kelas"])[["Kelas", "geometry"]]


bounds, COLOR_MAP = muat_meta(os.path.getmtime(f"{DATA_DIR}/peta_meta.json"))
bounds = [[-9.12, 110.85], [-4.92, 116.88]]
metrik = muat_metrik(os.path.getmtime(f"{DATA_DIR}/metrik.json"))
luas = muat_luas(os.path.getmtime(f"{DATA_DIR}/luas_per_kelas.csv"))

# Pengaman: Cek jika koordinat di peta_meta.json masih berupa meter UTM (> 180)
if abs(bounds[0][0]) > 90 or abs(bounds[0][1]) > 180:
    st.error("⚠️ Koordinat di `peta_meta.json` masih menggunakan satuan Meter (UTM), bukan Derajat (Latitude/Longitude)!")
    st.stop()

CM_DATA = metrik["confusion_matrix"]
N_POLIGON = sum(sum(baris) for baris in CM_DATA)
AKURASI = metrik["akurasi"] * 100
report = metrik["report"]

# Fleksibel membaca nama kelas di report (baik "Bangunan" maupun "Bangunan (0)")
PRECISION, RECALL, F1_SCORE = [], [], []
for idx, k in enumerate(CLASSES):
    kunci = k if k in report else f"{k} ({idx})"
    PRECISION.append(round(report[kunci]["precision"] * 100, 1))
    RECALL.append(round(report[kunci]["recall"] * 100, 1))
    F1_SCORE.append(round(report[kunci]["f1-score"] * 100, 1))

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================
st.sidebar.markdown("### 🌍 Spasial LULC")
st.sidebar.markdown("**Model:** Random Forest (Tekstur Spasial)")
st.sidebar.markdown("**Sensor:** Sentinel-2A (ESA CDSE)")
st.sidebar.markdown(f"**Akurasi Validasi Wilayah:** {AKURASI:.2f}%")
st.sidebar.divider()

st.sidebar.markdown("### 🏷️ 5 KELAS TUTUPAN LAHAN")
for kelas in CLASSES:
    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; margin-bottom:8px;">
            <span style="height:14px; width:14px; background-color:{COLOR_MAP[kelas]}; border-radius:50%; display:inline-block; margin-right:10px; border: 1px solid #ccc;"></span>
            <span style="font-size: 15px;">{kelas}</span>
        </div>
        """, unsafe_allow_html=True
    )

st.sidebar.divider()
st.sidebar.markdown("**👨‍💻 Dikembangkan Oleh:**")
st.sidebar.markdown("Muhammad Fathul Iman Wahid<br>NIM: 240411100180<br>Proyek Sains Data", unsafe_allow_html=True)

# ==============================================================================
# 4. HEADER & LUAS TIAP KELAS (INFORMASI HASIL PREDIKSI)
# ==============================================================================
st.title("🌍 Peta Tutupan Lahan Jawa Timur")
st.markdown("Hasil klasifikasi tutupan lahan seluruh Jawa Timur dari citra Sentinel-2 (Random Forest, sel prediksi 100 × 100 m).")

st.subheader("📐 Luas Tiap Kelas")
kolom_luas = st.columns(len(CLASSES))
for kolom, kelas in zip(kolom_luas, CLASSES):
    baris = luas[luas["Kelas"] == kelas].iloc[0]
    kolom.metric(label=kelas, value=f"{baris['Luas (ha)']:,.0f} ha")
    kolom.caption(f"{baris['Persen (%)']:.1f}% dari area terpetakan")

st.divider()

# ==============================================================================
# 5. PETA INTERAKTIF FOLIUM (HASIL PREDIKSI DI ATAS CITRA SATELIT)
# ==============================================================================
st.subheader("🗺️ Peta Hasil Klasifikasi")
st.markdown(
    "Gunakan **ikon kotak berlapis di pojok kanan atas peta** untuk mencentang atau "
    "menghapus centang kelas lahan (misalnya hanya **Pertanian**). Poligon sampel hasil "
    "digitasi tersedia sebagai layer tambahan (belum dicentang)."
)

selatan, barat = bounds[0]
utara, timur = bounds[1]
m = folium.Map(location=[(selatan + utara) / 2, (barat + timur) / 2], zoom_start=8, tiles=None)

# Basemap
folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google",
                 name="Google Satellite", overlay=False, control=True).add_to(m)
folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)

# Layer hasil prediksi: 1 kelas = 1 FeatureGroup
for kelas in CLASSES:
    grup = folium.FeatureGroup(name=f"Prediksi: {kelas}", show=True)
    ImageOverlay(image=f"{DATA_DIR}/kelas_{kelas}.png", bounds=bounds, opacity=0.85).add_to(grup)
    grup.add_to(m)

# Layer poligon sampel hasil digitasi
if GEOJSON_PATH:
    sampel = muat_sampel(GEOJSON_PATH, os.path.getmtime(GEOJSON_PATH))
    for kelas in CLASSES:
        sub = sampel[sampel["Kelas"] == kelas]
        if sub.empty:
            continue
        grup = folium.FeatureGroup(name=f"Sampel digitasi: {kelas}", show=False)
        folium.GeoJson(
            sub,
            style_function=lambda fitur, w=COLOR_MAP[kelas]: {
                "fillColor": w, "color": w, "weight": 1.5, "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(fields=["Kelas"], aliases=["Kelas lahan:"]),
        ).add_to(grup)
        grup.add_to(m)
else:
    st.warning("Catatan: File 'sampel_uts_jatim_fix.geojson' tidak ditemukan, layer sampel digitasi dilewati.")

m.fit_bounds(bounds)
folium.LayerControl(collapsed=False).add_to(m)

# Identitas mahasiswa melayang di peta
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

st_folium(m, use_container_width=True, height=650, returned_objects=[])

st.caption(
    "Catatan: prediksi dilakukan per sel 100 × 100 m sehingga sel di batas dua jenis lahan "
    "(misalnya mangrove dengan sawah, atau permukiman padat bervegetasi) hanya mendapat satu label. "
    "Peta ini tidak memiliki data acuan penuh, sehingga akurasinya diperkirakan mengikuti "
    "hasil validasi lintas wilayah di bawah."
)

st.divider()
# ==============================================================================
# 6. EVALUASI MODEL (DIBACA DARI metrik.json HASIL NOTEBOOK)
# ==============================================================================
st.subheader("📊 Evaluasi Kinerja Model")
st.markdown(f"Validasi lintas wilayah pada **{N_POLIGON} poligon**: akurasi **{AKURASI:.2f}%**.")

col_cm, col_bar = st.columns(2)

with col_cm:
    st.markdown(f"**Confusion Matrix ({N_POLIGON} Poligon, Validasi Lintas Wilayah)**")
    st.caption("Perbandingan kelas aktual lapangan vs hasil prediksi model.")
    fig_cm = px.imshow(
        CM_DATA, text_auto=True, color_continuous_scale="Teal", x=CLASSES, y=CLASSES,
        labels=dict(x="Hasil Prediksi Model", y="Kelas Ground Truth (Aktual)"),
    )
    fig_cm.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig_cm, width="stretch")  # <-- Sudah diganti dari use_container_width=True

with col_bar:
    st.markdown("**Kinerja Klasifikasi per Kelas (Persentase %)**")
    st.caption("Metrik Precision, Recall, dan F1-Score pada 5 kelas lahan.")
    fig_bar = go.Figure(data=[
        go.Bar(name="Precision", y=CLASSES, x=PRECISION, orientation="h", marker_color="#80cbc4"),
        go.Bar(name="Recall", y=CLASSES, x=RECALL, orientation="h", marker_color="#00897b"),
        go.Bar(name="F1-Score", y=CLASSES, x=F1_SCORE, orientation="h", marker_color="#004d40"),
    ])
    fig_bar.update_layout(barmode="group", margin=dict(l=20, r=20, t=20, b=20), height=350,
                          yaxis={"categoryorder": "array", "categoryarray": CLASSES[::-1]})
    st.plotly_chart(fig_bar, width="stretch")  # <-- Sudah diganti dari use_container_width=True
