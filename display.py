# import

import streamlit as st
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

# --- SentinelHub Config ---
config = SHConfig()
config.sh_client_id = st.secrets["sentinelhub"]["sh_client_id"]
config.sh_client_secret = st.secrets["sentinelhub"]["sh_client_secret"]
config.save()

# --- City BBoxes ---
CITIES = {
    "New York City, NY": BBox([-74.01, 40.70, -73.95, 40.76], crs=CRS.WGS84),
    "Los Angeles, CA": BBox([-118.27, 34.03, -118.20, 34.08], crs=CRS.WGS84),
    "Chicago, IL": BBox([-87.65, 41.87, -87.60, 41.91], crs=CRS.WGS84),
    "Houston, TX": BBox([-95.38, 29.74, -95.32, 29.79], crs=CRS.WGS84),
    "Phoenix, AZ": BBox([-112.08, 33.44, -112.02, 33.49], crs=CRS.WGS84),
    "Philadelphia, PA": BBox([-75.17, 39.94, -75.13, 39.98], crs=CRS.WGS84),
    "San Antonio, TX": BBox([-98.51, 29.41, -98.45, 29.47], crs=CRS.WGS84),
    "San Diego, CA": BBox([-117.17, 32.70, -117.12, 32.75], crs=CRS.WGS84),
    "Dallas, TX": BBox([-96.81, 32.77, -96.75, 32.82], crs=CRS.WGS84),
    "San Jose, CA": BBox([-121.90, 37.32, -121.84, 37.36], crs=CRS.WGS84),
    "Austin, TX": BBox([-97.75, 30.26, -97.70, 30.31], crs=CRS.WGS84),
    "Jacksonville, FL": BBox([-81.67, 30.31, -81.61, 30.36], crs=CRS.WGS84),
    "Fort Worth, TX": BBox([-97.34, 32.74, -97.28, 32.78], crs=CRS.WGS84),
    "Columbus, OH": BBox([-82.99, 39.95, -82.93, 40.00], crs=CRS.WGS84),
    "Charlotte, NC": BBox([-80.85, 35.21, -80.80, 35.25], crs=CRS.WGS84),
    "San Francisco, CA": BBox([-122.43, 37.76, -122.38, 37.80], crs=CRS.WGS84),
    "Indianapolis, IN": BBox([-86.17, 39.76, -86.12, 39.81], crs=CRS.WGS84),
    "Seattle, WA": BBox([-122.34, 47.60, -122.29, 47.65], crs=CRS.WGS84),
    "Denver, CO": BBox([-104.99, 39.73, -104.94, 39.77], crs=CRS.WGS84),
    "Washington, DC": BBox([-77.04, 38.88, -76.98, 38.92], crs=CRS.WGS84),
    "Boston, MA": BBox([-71.10, 42.34, -71.05, 42.38], crs=CRS.WGS84),
    "El Paso, TX": BBox([-106.49, 31.76, -106.44, 31.80], crs=CRS.WGS84),
    "Nashville, TN": BBox([-86.79, 36.15, -86.74, 36.20], crs=CRS.WGS84),
    "Detroit, MI": BBox([-83.09, 42.33, -83.04, 42.38], crs=CRS.WGS84),
    "Oklahoma City, OK": BBox([-97.54, 35.45, -97.48, 35.50], crs=CRS.WGS84),
    "Portland, OR": BBox([-122.68, 45.51, -122.62, 45.56], crs=CRS.WGS84),
    "Las Vegas, NV": BBox([-115.18, 36.16, -115.12, 36.20], crs=CRS.WGS84),
    "Memphis, TN": BBox([-90.06, 35.12, -90.00, 35.17], crs=CRS.WGS84),
    "Louisville, KY": BBox([-85.77, 38.23, -85.71, 38.27], crs=CRS.WGS84),
    "Baltimore, MD": BBox([-76.63, 39.28, -76.57, 39.33], crs=CRS.WGS84),
    "Milwaukee, WI": BBox([-87.93, 43.02, -87.87, 43.06], crs=CRS.WGS84),
    "Albuquerque, NM": BBox([-106.66, 35.07, -106.60, 35.11], crs=CRS.WGS84),
    "Tucson, AZ": BBox([-110.98, 32.21, -110.92, 32.26], crs=CRS.WGS84),
    "Fresno, CA": BBox([-119.80, 36.73, -119.74, 36.78], crs=CRS.WGS84),
    "Sacramento, CA": BBox([-121.50, 38.56, -121.44, 38.61], crs=CRS.WGS84),
    "Mesa, AZ": BBox([-111.84, 33.40, -111.78, 33.44], crs=CRS.WGS84),
    "Kansas City, MO": BBox([-94.60, 39.07, -94.54, 39.12], crs=CRS.WGS84),
    "Atlanta, GA": BBox([-84.40, 33.74, -84.34, 33.78], crs=CRS.WGS84),
    "Omaha, NE": BBox([-95.96, 41.25, -95.90, 41.29], crs=CRS.WGS84),
    "Colorado Springs, CO": BBox([-104.83, 38.82, -104.77, 38.86], crs=CRS.WGS84),
    "Raleigh, NC": BBox([-78.65, 35.77, -78.59, 35.81], crs=CRS.WGS84),
    "Miami, FL": BBox([-80.21, 25.76, -80.15, 25.80], crs=CRS.WGS84),
    "Long Beach, CA": BBox([-118.21, 33.77, -118.15, 33.82], crs=CRS.WGS84),
    "Virginia Beach, VA": BBox([-76.04, 36.82, -75.98, 36.86], crs=CRS.WGS84),
    "Oakland, CA": BBox([-122.28, 37.79, -122.22, 37.83], crs=CRS.WGS84),
    "Minneapolis, MN": BBox([-93.29, 44.96, -93.23, 45.00], crs=CRS.WGS84),
    "Bakersfield, CA": BBox([-119.04, 35.36, -118.98, 35.40], crs=CRS.WGS84),
    "Arlington, TX": BBox([-97.13, 32.71, -97.07, 32.75], crs=CRS.WGS84),
    "Wichita, KS": BBox([-97.34, 37.67, -97.28, 37.71], crs=CRS.WGS84),
    "Tampa, FL": BBox([-82.47, 27.94, -82.41, 27.98], crs=CRS.WGS84),
    "Tempe, AZ": BBox([-111.95, 33.38, -111.91, 33.42], crs=CRS.WGS84)
}


# --- Evalscript for NDVI ---
NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: {
      bands: 2,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi, sample.dataMask];
}
"""


# --- Streamlit UI ---
st.title("🌿 NDVI Visualizer")
city = st.selectbox("Select a city", list(CITIES.keys()))
min_date = date(2018, 1, 1)
max_date = date.today()
selected_date = st.date_input(
    "Select a date",
    value=date(2022, 1, 1),
    min_value=date(2018, 1, 1),
    max_value=date.today()
)


# --- Fetch NDVI for selected city & date ---
if st.button("Generate NDVI Image"):
    bbox = CITIES[city]
    start_date = selected_date - timedelta(days=7)
    end_date = selected_date + timedelta(days=7)
    date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    st.write(f"Fetching NDVI for **{city}** on **{selected_date}**...")

    request = SentinelHubRequest(
        evalscript=NDVI_EVALSCRIPT,
        input_data=[
    SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A,
        time_interval=date_range,
        mosaicking_order='leastCC',
        maxcc=0.2  # 20% cloud cover
    )
],

        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=bbox,
        size=(512, 512),
        config=config
    )

    try:
      
        data = request.get_data()[0]
        ndvi_image = data[:, :, 0]
        mask = data[:, :, 1]

        if np.all(mask == 0):
            st.warning(f"It was too cloudy around **{selected_date}**, couldn't get the satellite image.")
        else:
            ndvi_image[mask == 0] = np.nan
            normalized_ndvi = np.clip((ndvi_image + 1) / 2, 0, 1)  # Normalize NDVI from [-1, 1] to [0, 1]
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(normalized_ndvi, cmap="YlGn", vmin=0, vmax=1)  # Show as 0 to 1
            plt.colorbar(im, ax=ax, label="NDVI (Normalized)")

            ax.axis("off")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to retrieve image: {e}")
