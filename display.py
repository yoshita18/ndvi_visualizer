import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
from sklearn.ensemble import IsolationForest

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


# --- Evalscript for NDVI, SAVI, and EVI ---
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "B02"],
    output: {
      bands: 3,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  // NDVI
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  
  // SAVI
  let savi = ((sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.5)) * 1.5;

  // EVI
  let evi = 2.5 * (sample.B08 - sample.B04) / (sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 10000);

  return [ndvi, savi, evi];
}
"""

# --- Function to fetch data ---
def fetch_data_for_city(bbox, start_date, end_date):
    date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=date_range,
            mosaicking_order='leastCC',  # Fetch least cloud cover images
            maxcc=0.2  # 20% cloud coverage
        )],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=bbox,
        size=(512, 512),
        config=config
    )

    try:
        data = request.get_data()[0]  # Get the data for that time period
        ndvi = data[:, :, 0]
        savi = data[:, :, 1]
        evi = data[:, :, 2]
        return ndvi, savi, evi
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

# --- Time Series Analysis and Anomaly Detection ---
def detect_anomalies(data, contamination=0.05):
    model = IsolationForest(contamination=contamination)
    anomalies = model.fit_predict(data)
    return anomalies

# --- Streamlit UI ---
st.title("🌾 Crop Disease Monitoring using NDVI")

# Add options to choose between visualizing NDVI, SAVI, EVI or anomaly detection
option = st.radio("Select an option", ["Visualize NDVI", "Detect Anomalies"])

# Select city and date range
# city = st.selectbox("Select a city", list(CITIES.keys()))

if option == "Visualize NDVI":
    city = st.selectbox("Select a city", list(CITIES.keys()), key="city_selectbox")
    
    # Min and Max date setup
    min_date = date(2018, 1, 1)
    max_date = date.today()
    
    # Add a unique key to the date_input as well
    selected_date = st.date_input(
        "Select a date",
        value=date(2022, 1, 1),
        min_value=date(2018, 1, 1),
        max_value=date.today(),
        key="date_input"
    )
    
    # Get the bounding box for the selected city
    bbox = CITIES[city]
    
    # Function to fetch data and check if it is valid (not all zero mask)
    def fetch_valid_data(bbox, selected_date):
        for delta in range(-7, 8):  # Trying ±7 days
            current_date = selected_date + timedelta(days=delta)
            start_date = current_date - timedelta(days=7)
            end_date = current_date + timedelta(days=7)
            date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            # Request to fetch data from SentinelHub
            request = SentinelHubRequest(
                evalscript=EVALSCRIPT,
                input_data=[SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=date_range,
                    mosaicking_order='leastCC',
                    maxcc=0.2  # 20% cloud cover
                )],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=(512, 512),
                config=config
            )

            try:
                # Fetch the data
                data = request.get_data()[0]
                ndvi_image = data[:, :, 0]
                mask = data[:, :, 1]

                if np.all(mask == 0):  # If all elements in mask are zero
                    continue  # Try the next date if all elements are zero
                else:
                    # Normalize the NDVI values and plot the image
                    ndvi_image[mask == 0] = np.nan
                    normalized_ndvi = np.clip((ndvi_image + 1) / 2, 0, 1)  # Normalize NDVI from [-1, 1] to [0, 1]
                    return normalized_ndvi, current_date  # Return the valid data and corresponding date

            except Exception as e:
                continue  # If data fetching failed, try the next date

        # If no valid data was found after ±7 days, return None
        return None, None

    # Fetch data for selected city and date
    normalized_ndvi, valid_date = fetch_valid_data(bbox, selected_date)
    
    if normalized_ndvi is not None:
        # Plot the valid NDVI data
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(normalized_ndvi, cmap="YlGn", vmin=0, vmax=1)  # Show as 0 to 1
        plt.colorbar(im, ax=ax, label="NDVI (Normalized)")

        ax.axis("off")
        st.write(f"Fetched NDVI for **{city}** on **{valid_date}**.")
        st.pyplot(fig)

    else:
        st.warning(f"It was too cloudy around **{selected_date}** and the surrounding dates. Couldn't get the satellite image.")



elif option == "Detect Anomalies":
    biweekly_ndvi_dataset = pd.read_csv('biweekly_ndvi_dataset.csv')

    # Ensure the 'Start Date' column is in datetime format
    biweekly_ndvi_dataset['Start Date'] = pd.to_datetime(biweekly_ndvi_dataset['Start Date'])

    # Set 'Start Date' as the index
    biweekly_ndvi_dataset.set_index('Start Date', inplace=True)

    # st.title("🌾 Crop Disease Monitoring using NDVI, SAVI, and EVI")

    # --- City Selection ---
    city = st.selectbox("Select a city", list(biweekly_ndvi_dataset['City'].unique()))
    
    min_date = date(2018, 1, 1)  # The earliest date that can be selected
    max_date = date.today()  # The latest date that can be selected

    # Start Date with restriction
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,  # Ensure start date cannot be before 2018-01-01
        max_value=max_date  # Ensure start date cannot be after today's date
    )

    # End Date with restriction
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=start_date,  # Ensure that the end date cannot be before the start date
        max_value=max_date  # Ensure end date cannot be after today's date
    )
    
    if (end_date - start_date).days < 365:
        st.write("**Tip:** To get a clear view of anomalies, make sure the range of dates is at least a year.")

    # Filter the dataset based on the selected city and date range
    city_data = biweekly_ndvi_dataset[(biweekly_ndvi_dataset['City'] == city) & 
                                    (biweekly_ndvi_dataset.index >= pd.to_datetime(start_date)) & 
                                    (biweekly_ndvi_dataset.index <= pd.to_datetime(end_date))]

    # --- Time Series Analysis and Anomaly Detection ---
    if city_data.empty:
        st.write(f"No data available for **{city}** in the selected date range.")
    else:
        # Prepare data for anomaly detection (only the 'NDVI' column for now)
        data = city_data[['NDVI']]

        # Detect anomalies
        anomalies = detect_anomalies(data[['NDVI']].values)

        # Add anomalies to the DataFrame
        city_data['anomaly'] = anomalies

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot NDVI with anomalies highlighted
        ax.plot(city_data.index, city_data['NDVI'], label='NDVI', color='blue')
        ax.scatter(city_data.index[city_data['anomaly'] == -1], city_data['NDVI'][city_data['anomaly'] == -1], 
                color='red', label='Anomalies', zorder=5)

        # Adding title, labels, legend, and grid
        ax.set_title(f'{city} NDVI Time Series with Anomalies Detected')
        ax.set_xlabel('Date')
        ax.set_ylabel('NDVI')
        ax.legend()
        ax.grid(True)

        # Display the plot
        st.pyplot(fig)

        # Show the anomalies in a table
        st.write("Anomalies Detected:")
        st.dataframe(city_data[city_data['anomaly'] == -1])

        # Alert system if anomalies are detected
        if np.any(anomalies == -1):
            st.warning(f"Alert: Crop disease detected! Anomalies have been detected in {city}'s crop health indicators.")
        else:
            st.success(f"No anomalies detected for {city}.")
        
