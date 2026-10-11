
import os
import unicodedata

import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="Dashboard LULC Jawa Timur",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        .dashboard-subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 16px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌏 Dashboard Interaktif Tutupan Lahan (LULC) Jawa Timur")
st.markdown(
    '<p class="dashboard-subtitle">'
    "Visualisasi dan eksplorasi data poligon Land Use / Land Cover "
    "secara interaktif."
    "</p>",
    unsafe_allow_html=True,
)


# =========================================================
# 2. KONFIGURASI DATA DAN KELAS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sampel_uts_jatim_fix.geojson")

CLASS_COLORS = {
    "Bangunan": "#E53935",  # Merah
    "Pertanian": "#8BC34A",  # Hijau muda
    "Air": "#2196F3",       # Biru
    "Hutan": "#1B5E20",     # Hijau tua
    "Mangrove": "#8E24AA",  # Ungu
}

CLASS_CODES = {
    "b": "Bangunan",
    "p": "Pertanian",
    "s": "Pertanian",
    "a": "Air",
    "h": "Hutan",
    "m": "Mangrove",
}

CLASS_ORDER = [
    "Bangunan",
    "Pertanian",
    "Air",
    "Hutan",
    "Mangrove",
]


def normalize_text(value):
    """Membersihkan teks untuk mempermudah identifikasi kelas."""
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    return text.strip().lower()


def classify_land_cover(kelas_value, nama_value):
    """
    Menentukan kelas tutupan lahan.
    Prioritas: kolom Kelas yang dikenali, lalu kolom Nama.
    """

    # Nama kelas lengkap yang mungkin ditemukan dalam data
    full_names = {
        "bangunan": "Bangunan",
        "pertanian": "Pertanian",
        "air": "Air",
        "hutan": "Hutan",
        "mangrove": "Mangrove",
    }

    # Periksa kolom Kelas terlebih dahulu
    kelas_text = normalize_text(kelas_value)

    if kelas_text in CLASS_CODES:
        return CLASS_CODES[kelas_text]

    if kelas_text in full_names:
        return full_names[kelas_text]

    # Jika Kelas kosong atau tidak dikenali, gunakan Nama
    nama_text = normalize_text(nama_value)

    if nama_text:
        first_letter = nama_text[0]

        if first_letter in CLASS_CODES:
            return CLASS_CODES[first_letter]

        # Mendukung nilai Nama yang berupa nama kelas lengkap
        if nama_text in full_names:
            return full_names[nama_text]

    return "Tidak Diketahui"


# =========================================================
# 3. MEMBACA GEOJSON SECARA OTOMATIS (READ-ONLY)
# =========================================================

@st.cache_data(show_spinner="Memuat data geospasial...")
def load_geojson(file_path, modified_time):
    """
    Membaca GeoJSON tanpa mengubah file sumber.
    modified_time digunakan agar cache diperbarui
    ketika file sumber berubah.
    """
    del modified_time

    gdf = gpd.read_file(file_path)

    if gdf.empty:
        raise ValueError("File GeoJSON tidak memiliki fitur.")

    if gdf.crs is None:
        # GeoJSON umumnya menggunakan koordinat WGS84.
        # Pastikan CRS sumber benar sebelum menetapkannya.
        gdf = gdf.set_crs("EPSG:4326")

    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Pastikan kolom yang dibutuhkan tersedia
    if "Kelas" not in gdf.columns:
        gdf["Kelas"] = None

    if "Nama" not in gdf.columns:
        gdf["Nama"] = None

    # Klasifikasi setiap fitur
    gdf["kelas_dashboard"] = gdf.apply(
        lambda row: classify_land_cover(
            row["Kelas"],
            row["Nama"],
        ),
        axis=1,
    )

    # Hanya tampilkan fitur dengan geometri yang tersedia
    gdf = gdf.loc[
        gdf.geometry.notna() & ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError("Tidak ditemukan geometri poligon yang valid.")

    return gdf


if not os.path.exists(DATA_PATH):
    st.error(
        "File GeoJSON tidak ditemukan. Pastikan "
        "`sampel_uts_jatim_fix.geojson` berada di folder yang sama "
        "dengan `app.py`."
    )
    st.stop()

try:
    gdf = load_geojson(
        DATA_PATH,
        os.path.getmtime(DATA_PATH),
    )
except Exception as exc:
    st.error(f"Gagal membaca data geospasial: {exc}")
    st.stop()


# =========================================================
# 4. FILTER KELAS TUTUPAN LAHAN
# =========================================================

available_classes = [
    kelas
    for kelas in CLASS_ORDER
    if kelas in gdf["kelas_dashboard"].unique()
]

unknown_count = int(
    (gdf["kelas_dashboard"] == "Tidak Diketahui").sum()
)

if unknown_count:
    available_classes.append("Tidak Diketahui")

st.subheader("Filter Tutupan Lahan")

selected_classes = st.multiselect(
    "Pilih kategori yang ingin ditampilkan pada peta:",
    options=available_classes,
    default=available_classes,
    help="Pilih satu atau beberapa kelas untuk memfilter poligon.",
)

filtered_gdf = gdf[
    gdf["kelas_dashboard"].isin(selected_classes)
].copy()


# =========================================================
# 5. KARTU METRIK
# =========================================================

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric(
        label="Total Sampel Poligon",
        value=f"{len(filtered_gdf):,}",
        help="Jumlah poligon yang sesuai dengan filter aktif.",
    )

with metric_col2:
    active_categories = filtered_gdf["kelas_dashboard"].nunique()

    st.metric(
        label="Kategori Aktif",
        value=f"{active_categories}",
        help="Jumlah kategori yang benar-benar muncul pada hasil filter.",
    )

st.caption(
    f"Total data terbaca: {len(gdf):,} poligon | "
    f"Kategori tersedia: {len(available_classes)}"
)


# =========================================================
# 6. MEMBUAT PETA FOLIUM
# =========================================================

st.subheader("Peta Sebaran Tutupan Lahan")

# Tentukan titik tengah berdasarkan seluruh data agar
# tampilan peta tidak berubah ekstrem ketika filter diganti.
bounds = gdf.total_bounds
minx, miny, maxx, maxy = bounds

center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

lulc_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles=None,
    control_scale=True,
    dragging=False,
    scrollWheelZoom=True,
    zoomControl=True,
    doubleClickZoom=True,
    touchZoom=True,
)

# Basemap Google Satellite
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Map data © Google",
    name="Google Satellite",
    overlay=False,
    control=False,
    show=True,
).add_to(lulc_map)


# =========================================================
# 7. STYLE POLIGON DAN TOOLTIP
# =========================================================

def style_polygon(feature):
    """Mengatur warna isi dan garis tepi poligon."""

    kelas = feature["properties"].get(
        "kelas_dashboard",
        "Tidak Diketahui",
    )

    color = CLASS_COLORS.get(kelas, "#9E9E9E")

    return {
        "fillColor": color,
        "color": color,
        "weight": 1.5,
        "opacity": 0.6,
        "fillOpacity": 0.4,
    }


if not filtered_gdf.empty:
    tooltip_fields = ["kelas_dashboard"]
    tooltip_aliases = ["Kategori Tutupan Lahan"]

    # Tambahkan informasi Nama jika tersedia dan terisi
    if "Nama" in filtered_gdf.columns:
        if filtered_gdf["Nama"].notna().any():
            tooltip_fields.insert(0, "Nama")
            tooltip_aliases.insert(0, "Nama Poligon")

    folium.GeoJson(
        data=filtered_gdf.to_json(),
        name="Tutupan Lahan",
        style_function=style_polygon,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            sticky=True,
            localize=True,
        ),
    ).add_to(lulc_map)

else:
    st.info(
        "Tidak ada poligon untuk kategori yang dipilih. "
        "Silakan aktifkan setidaknya satu kategori."
    )


# =========================================================
# 8. LEGENDA PETA
# =========================================================

legend_items = [
    (kelas, CLASS_COLORS[kelas])
    for kelas in CLASS_ORDER
    if kelas in available_classes and kelas in selected_classes
]

if "Tidak Diketahui" in selected_classes:
    legend_items.append(("Tidak Diketahui", "#9E9E9E"))

legend_html = """
<div style="
    position: fixed;
    bottom: 25px;
    left: 25px;
    z-index: 9999;
    background: white;
    padding: 12px 16px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-size: 13px;
    color: #1f2937;
">
    <b style="font-size:14px;">Legenda Tutupan Lahan</b>
    <div style="margin-top:8px;">
"""

for kelas, color in legend_items:
    legend_html += f"""
        <div style="display:flex; align-items:center; margin:5px 0;">
            <span style="
                display:inline-block;
                width:13px;
                height:13px;
                background:{color};
                border:1px solid {color};
                margin-right:8px;
                border-radius:2px;
            "></span>
            <span>{kelas}</span>
        </div>
    """

legend_html += "</div></div>"

lulc_map.get_root().html.add_child(
    folium.Element(legend_html)
)


# =========================================================
# 9. MENAMPILKAN PETA DI STREAMLIT
# =========================================================

st_folium(
    lulc_map,
    use_container_width=True,
    height=650,
    returned_objects=[],
)

st.caption(
    "Sumber data: sampel_uts_jatim_fix.geojson. "
    "Peta menggunakan basemap Google Satellite. "
    "Data sumber hanya dibaca dan tidak dimodifikasi oleh aplikasi."
)
