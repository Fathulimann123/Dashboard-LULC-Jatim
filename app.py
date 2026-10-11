
from pathlib import Path
import html
import unicodedata

import geopandas as gpd
import joblib
import folium
import numpy as np
import pandas as pd
import streamlit as st

from branca.element import Element
from streamlit_folium import st_folium


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="Dashboard LULC Jawa Timur",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = BASE_DIR / "sampel_uts_jatim_fix.geojson"
MODEL_PATH = BASE_DIR / "model_LULC_jatim_texture.joblib"

CLASS_ORDER = [
    "Bangunan",
    "Pertanian",
    "Air",
    "Hutan",
    "Mangrove",
]

CLASS_COLORS = {
    "Bangunan": "#E53935",
    "Pertanian": "#FBC02D",
    "Air": "#2196F3",
    "Hutan": "#1B5E20",
    "Mangrove": "#8E24AA",
    "Tidak diketahui": "#9E9E9E",
}

CLASS_CODES = {
    "b": "Bangunan",
    "p": "Pertanian",
    "s": "Pertanian",
    "a": "Air",
    "h": "Hutan",
    "m": "Mangrove",
}

# Hasil evaluasi sesuai angka yang diberikan dalam blueprint.
# Ganti hanya jika laporan evaluasi final berbeda.
MODEL_ACCURACY = 0.7584

F1_SCORES = {
    "Bangunan": 0.95,
    "Pertanian": 0.47,
    "Air": 0.88,
    "Hutan": 0.74,
    "Mangrove": 0.77,
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.2);
        border-radius: 12px;
        padding: 14px;
    }

    h1, h2, h3 {
        letter-spacing: -0.3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dashboard Interaktif Tutupan Lahan (LULC) Jawa Timur")
st.caption(
    "Eksplorasi data spasial, hasil evaluasi klasifikasi, "
    "dan distribusi kategori tutupan lahan."
)


# =========================================================
# 2. FUNGSI UTILITAS
# =========================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    return unicodedata.normalize(
        "NFKC", str(value)
    ).strip().lower()


def classify_land_cover(kelas, nama):
    """
    Prioritas klasifikasi:
    1. Nilai kolom Kelas yang dikenali.
    2. Huruf pertama kolom Nama.
    """

    full_names = {
        "bangunan": "Bangunan",
        "pertanian": "Pertanian",
        "air": "Air",
        "hutan": "Hutan",
        "mangrove": "Mangrove",
    }

    kelas_text = normalize_text(kelas)

    if kelas_text in CLASS_CODES:
        return CLASS_CODES[kelas_text]

    if kelas_text in full_names:
        return full_names[kelas_text]

    nama_text = normalize_text(nama)

    if nama_text:
        first_letter = nama_text[0]

        if first_letter in CLASS_CODES:
            return CLASS_CODES[first_letter]

        if nama_text in full_names:
            return full_names[nama_text]

    return "Tidak diketahui"


# =========================================================
# 3. MEMUAT DATA DAN MODEL
# =========================================================

@st.cache_data(show_spinner="Memuat data geospasial...")
def load_geojson(path, modified_time):
    del modified_time

    data = gpd.read_file(path)

    if data.empty:
        raise ValueError("GeoJSON tidak memiliki fitur.")

    if data.crs is None:
        # GeoJSON standar menggunakan koordinat WGS84.
        data = data.set_crs("EPSG:4326")
    else:
        data = data.to_crs("EPSG:4326")

    if "Kelas" not in data.columns:
        data["Kelas"] = None

    if "Nama" not in data.columns:
        data["Nama"] = None

    data["kelas_dashboard"] = data.apply(
        lambda row: classify_land_cover(
            row["Kelas"],
            row["Nama"],
        ),
        axis=1,
    )

    data = data.loc[
        data.geometry.notna()
        & ~data.geometry.is_empty
    ].copy()

    if data.empty:
        raise ValueError("Tidak ditemukan geometri valid.")

    return data


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def load_model(path, modified_time):
    del modified_time
    return joblib.load(path)


# Validasi file sebelum pemrosesan
if not GEOJSON_PATH.exists():
    st.error(
        f"GeoJSON tidak ditemukan: {GEOJSON_PATH.name}. "
        "Pastikan file berada satu folder dengan app.py."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        f"Model tidak ditemukan: {MODEL_PATH.name}. "
        "Pastikan file model berada satu folder dengan app.py."
    )
    st.stop()

try:
    gdf = load_geojson(
        str(GEOJSON_PATH),
        GEOJSON_PATH.stat().st_mtime,
    )

    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

except Exception as exc:
    st.error(f"Gagal memuat data atau model: {exc}")
    st.stop()


# =========================================================
# 4. SIDEBAR: PERFORMA MODEL
# =========================================================

st.sidebar.title("Panel Model")
st.sidebar.caption("Ringkasan evaluasi klasifikasi")

st.sidebar.metric(
    "Akurasi Validasi Wilayah",
    f"{MODEL_ACCURACY:.2%}",
    help="Nilai dari hasil evaluasi validasi silang spasial."
)

st.sidebar.divider()
st.sidebar.subheader("Feature Importance")

# PERBAIKAN: Membaca langsung dari paket dictionary joblib Anda
if isinstance(model, dict) and "fitur" in model and "fitur_importance" in model:
    importance_df = pd.DataFrame({
        "Fitur": model["fitur"],
        "Importance": model["fitur_importance"]
    }).sort_values("Importance", ascending=False)
    
    st.sidebar.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True,
        height=260,
    )
else:
    st.sidebar.info("Model tidak menyediakan feature_importances_.")

# =========================================================
# 5. METRIK F1-SCORE PER KELAS
# =========================================================

st.subheader("Ringkasan Performa Klasifikasi")

metric_cols = st.columns(5)

for col, kelas in zip(metric_cols, CLASS_ORDER):
    with col:
        st.metric(
            label=f"F1-Score {kelas}",
            value=f"{F1_SCORES[kelas]:.2f}",
        )

st.info(
    "Interpretasi: skor F1 menunjukkan keseimbangan "
    "precision dan recall. Perbedaan skor antarkelas "
    "menunjukkan bahwa performa model tidak merata. "
    "Khusus kelas Pertanian dan Hutan, hasil evaluasi "
    "perlu diperhatikan lebih lanjut."
)


# =========================================================
# 6. FILTER DATA SPASIAL
# =========================================================

st.divider()
st.subheader("Eksplorasi Data Spasial")

# Deteksi kolom klaster jika tersedia.
# Tidak mengasumsikan semua GeoJSON memiliki kolom DBSCAN.
cluster_candidates = [
    col for col in gdf.columns
    if any(
        key in col.lower()
        for key in ["cluster", "dbscan", "klaster"]
    )
    and col != "geometry"
]

filter_col, info_col = st.columns([3, 1])

with filter_col:
    selected_classes = st.multiselect(
        "Filter kelas tutupan lahan",
        options=[
            kelas
            for kelas in CLASS_ORDER
            if kelas in gdf["kelas_dashboard"].unique()
        ],
        default=[
            kelas
            for kelas in CLASS_ORDER
            if kelas in gdf["kelas_dashboard"].unique()
        ],
        key="filter_kelas",
    )

with info_col:
    st.metric(
        "Total Poligon",
        f"{len(gdf):,}",
    )

filtered_gdf = gdf[
    gdf["kelas_dashboard"].isin(selected_classes)
].copy()

selected_cluster_col = None

if cluster_candidates:
    selected_cluster_col = st.selectbox(
        "Kolom wilayah/klaster",
        options=cluster_candidates,
        help="Pilih kolom klaster yang benar dari GeoJSON.",
    )

    cluster_values = sorted(
        filtered_gdf[selected_cluster_col]
        .dropna()
        .unique()
        .tolist(),
        key=str,
    )

    selected_clusters = st.multiselect(
        "Filter klaster",
        options=cluster_values,
        default=cluster_values,
    )

    filtered_gdf = filtered_gdf[
        filtered_gdf[selected_cluster_col].isin(
            selected_clusters
        )
    ].copy()


# =========================================================
# 7. PETA INTERAKTIF
# =========================================================

st.divider()
st.subheader("Peta Tutupan Lahan Jawa Timur")

map_col, legend_col = st.columns(
    [4, 1],
    gap="medium",
)

with legend_col:
    st.markdown("#### Legenda")

    for kelas in CLASS_ORDER:
        if kelas in selected_classes:
            color = CLASS_COLORS[kelas]

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;
                            gap:8px;margin:10px 0;">
                    <span style="display:inline-block;
                                 width:15px;height:15px;
                                 border-radius:3px;
                                 background:{color};"></span>
                    <span>{html.escape(kelas)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        f"{len(filtered_gdf):,} poligon ditampilkan."
    )


with map_col:
    # Pusat awal Jawa Timur
    m = folium.Map(
        location=[-7.6, 112.5],
        zoom_start=8,
        tiles=None,
        control_scale=True,
        dragging=False,
        scrollWheelZoom=True,
        doubleClickZoom=True,
        touchZoom=True,
        zoomControl=True,
        keyboard=False,
        preferCanvas=True,
    )

    # Basemap OpenStreetMap
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        show=True,
    ).add_to(m)

    # Citra satelit Esri
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        name="Esri World Imagery",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Fit seluruh poligon terfilter.
    # Jika tidak ada data, pusat awal tetap Jawa Timur.
    if not filtered_gdf.empty:
        minx, miny, maxx, maxy = (
            filtered_gdf.total_bounds
        )

        m.fit_bounds(
            [[miny, minx], [maxy, maxx]],
            padding=(20, 20),
        )

        def polygon_style(feature):
            kelas = feature["properties"].get(
                "kelas_dashboard",
                "Tidak diketahui",
            )

            color = CLASS_COLORS.get(
                kelas,
                CLASS_COLORS["Tidak diketahui"],
            )

            return {
                "color": color,
                "fillColor": color,
                "weight": 1.5,
                "opacity": 0.6,
                "fillOpacity": 0.4,
            }

        tooltip_fields = ["kelas_dashboard"]
        tooltip_aliases = ["Kelas:"]

        if (
            "Nama" in filtered_gdf.columns
            and filtered_gdf["Nama"].notna().any()
        ):
            tooltip_fields.insert(0, "Nama")
            tooltip_aliases.insert(0, "Nama: ")

        folium.GeoJson(
            filtered_gdf.to_json(),
            name="Poligon LULC",
            style_function=polygon_style,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                sticky=True,
            ),
        ).add_to(m)

    else:
        st.warning(
            "Tidak ada poligon yang cocok dengan filter."
        )

    folium.LayerControl(
        position="topright",
        collapsed=False,
    ).add_to(m)

    # Kunci dragging melalui Leaflet setelah map diinisialisasi.
    # Zoom tetap aktif.
    map_name = m.get_name()

    m.get_root().script.add_child(
        Element(
            f"""
            {map_name}.dragging.disable();
            {map_name}.scrollWheelZoom.enable();
            {map_name}.doubleClickZoom.enable();
            {map_name}.touchZoom.enable();
            """
        )
    )

    st_folium(
        m,
        key="lulc_map",
        height=650,
        use_container_width=True,
        returned_objects=[],
    )


# =========================================================
# 8. TABEL ATRIBUT
# =========================================================

st.divider()
st.subheader("Inspeksi Atribut Poligon")

st.caption(
    "Tabel mengikuti filter kelas dan klaster yang aktif. "
    "Kolom geometri tidak ditampilkan."
)

attribute_df = pd.DataFrame(
    filtered_gdf.drop(
        columns=["geometry"],
        errors="ignore",
    )
)

st.dataframe(
    attribute_df,
    use_container_width=True,
    hide_index=True,
    height=350,
)

st.caption(
    "Sumber data: sampel_uts_jatim_fix.geojson. "
    "Aplikasi hanya membaca file sumber dan tidak "
    "menyimpan perubahan ke GeoJSON."
)
