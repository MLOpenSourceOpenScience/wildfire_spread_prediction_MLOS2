import netCDF4 as nc
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

# Earth's radius in kilometers
R = 6371.0

# Function to convert degrees to radians
def deg2rad(degrees):
    return degrees * np.pi / 180

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Function to filter points close to any centroid
def filter_points(combined_data, centroids, max_distance_km=20):
    # Create a KD-tree for fast spatial searching
    tree = cKDTree(combined_data[:, [2, 3]])  # Adjust indices for lat and lon
    filtered_indices = set()
    
    for centroid in centroids:
        # Convert centroid location to radians for consistency in haversine function use
        centroid_lat_rad, centroid_lon_rad = map(deg2rad, centroid)
        # Query the tree for points within `max_distance_km`
        for i in range(len(combined_data)):
            point_lat_rad, point_lon_rad = map(deg2rad, combined_data[i, [2, 3]])
            distance = haversine(centroid_lat_rad, centroid_lon_rad, point_lat_rad, point_lon_rad)
            if distance <= max_distance_km:
                filtered_indices.add(i)
    
    # Convert the set to a sorted list and use it to filter the combined_data
    return combined_data[list(filtered_indices)]

print("Reading in fire data...")
gdf = gpd.read_file('dataframe/dataframe.shp')
gdf.dropna(inplace=True)
print("Finished reading in fire data")
discretized_lat_lon = False
variables_t = {
        "AvgSurfT_tavg",
        "Rainf_tavg",
        "TVeg_tavg",
        "Wind_f_tavg",
        "Tair_f_tavg",
        "Qair_f_tavg",
        "SoilMoi00_10cm_tavg",
        "SoilTemp00_10cm_tavg",
    }
# Create an empty DataFrame to store the data
df_columns = ["time", "date", "lat", "lon"] + list(variables_t)
df = pd.DataFrame(columns=df_columns)

start_date = "WLDAS_NOAHMP001_DA1_20170717"
end_date = "WLDAS_NOAHMP001_DA1_20170718"
folder_dir = "../dataset/WLDAS"
for fid, file in tqdm(enumerate(sorted(os.listdir(folder_dir))), total=len(os.listdir(folder_dir))):

    # actual start and end dates of fire are 2017-07-16 and 2017-08-24

    if file[:len(start_date)] < start_date or file[:len(end_date)] > end_date:
        continue

    if not file.endswith('.nc'):
        continue

    file_path = os.path.join(folder_dir, file)
    ds = nc.Dataset(file_path)
    
    

    time = ds.variables["time"][:].astype(float) 
    variables = {
        "AvgSurfT_tavg",
        "Rainf_tavg",
        "TVeg_tavg",
        "Wind_f_tavg",
        "Tair_f_tavg",
        "Qair_f_tavg",
        "SoilMoi00_10cm_tavg",
        "SoilTemp00_10cm_tavg",
    }

    data_list = []
    for var in variables:
        data = ds.variables[var][:].astype(float)
        data_list.append(data)
    data_array = np.array(data_list)
    lat = ds.variables["lat"][:].astype(float)
    lon = ds.variables["lon"][:].astype(float)

    if not discretized_lat_lon:
        gdf["lat_bins"] = pd.cut(gdf["LATITUDE"], bins=lat, precision=10, labels=False)  # remove latitudes and longitudes that are not in western North America
        gdf["lon_bins"] = pd.cut(gdf["LONGITUDE"], bins=lon, precision=10, labels=False)
        gdf["lat_bins"] = len(lat) - gdf["lat_bins"] - 1
        gdf.dropna(inplace=True)
        discretized_lat_lon = True


    date = f"{file[20:24]}-{file[24:26]}-{file[26:28]}"
    
    # Check if the lat and lon values are within the bounds of your dataset
    mask_lat = (lat <= gdf["LATITUDE"].min()) & (lat >= gdf["LATITUDE"].max())
    mask_lon = (lon <= gdf["LONGITUDE"].min()) & (lon >= gdf["LONGITUDE"].max())

    # Filter lat and lon based on the mask
    filtered_lat = lat[mask_lat]
    filtered_lon = lon[mask_lon]

    fires = gdf[gdf["ACQ_DATE"].str.contains(date, na=False)]

    fire_array = np.zeros((len(lat), len(lon)), dtype=np.uint8)
    fire_array[fires["lat_bins"].astype(int), fires["lon_bins"].astype(int)] = 1
    
    # Append data to the DataFrame
    data_array = data_array[:,0,:,:]
    lat = lat.reshape(1,-1,1)
    lon = lon.reshape(1,1,-1)
    time = time.reshape(1,1,1)
    date = np.array([date]*data_array.shape[1]*data_array.shape[2]).reshape((1, data_array.shape[1], data_array.shape[2]))
    lat = np.tile(lat, (1, 1, data.shape[2]))
    lon = np.tile(lon, (1, data.shape[1], 1))
    time = np.tile(time, (1, data.shape[1], data.shape[2]))
    fire_array = fire_array.reshape(1, data_array.shape[1], data_array.shape[2])
    combined_data = np.concatenate((time, date, lat, lon, data_array, fire_array), axis=0)
    combined_data = combined_data.reshape(combined_data.shape[0], -1)
    print(combined_data.shape) 
    fire_points = combined_data[combined_data[:,-1]==1]
    fire_points = fire_points.reshape(-1, combined_data.shape[1])
    
    # Clustering fire points
    fire_lat_lon = fire_points[:, 2:4]  # Adjust these indices based on your data
    dbscan = DBSCAN(eps=0.05, min_samples=10, metric=lambda u, v: haversine(u[0], u[1], v[0], v[1]))
    clusters = dbscan.fit_predict(fire_lat_lon)
    # Calculate centroids of clusters
    centroids = []
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:
            index = clusters == cluster_id
            centroid_lat = np.mean(fire_lat_lon[index, 0])
            centroid_lon = np.mean(fire_lat_lon[index, 1])
            centroids.append((centroid_lat, centroid_lon))
    
    
    # Create a mask for points within 20 km from any centroid
    proximity_mask = np.zeros(len(combined_data), dtype=int)

    for centroid in centroids:
        # Each centroid processed
        centroid_lat, centroid_lon = centroid
        for i, point in enumerate(combined_data):
            point_lat, point_lon = point[2], point[3]  # Adjust indices if necessary
            if haversine(centroid_lat, centroid_lon, point_lat, point_lon) <= 20:
                proximity_mask[i] = 1

    # Add the mask as a new column to combined_data
    combined_data_with_mask = np.hstack((combined_data, proximity_mask[:, None]))

    print(combined_data_with_mask.shape)  # This will show the original number of rows and one additional column



    '''
        Convert the array called combined_data to DataFrame here!!! 
        combined_data is a 2D array with shape (12, ~10 million). Where 12 is the number of variables and 10 million is the number of pixels in each map.
        The variables are time, date, lat, lon, and the 8 WLDAS variables in the same order as variables dictionary on line 48
    '''
    variable_names = ['time', 'date', 'lat', 'lon', 'AvgSurfT_tavg', 'Rainf_tavg', 'TVeg_tavg', 'Wind_f_tavg', 'Tair_f_tavg', 'Qair_f_tavg', 'SoilMoi00_10cm_tavg', 'SoilTemp00_10cm_tavg', 'fire'] 

    # Create a DataFrame
    df = pd.DataFrame(data=combined_data.T, columns=variable_names)
    # Filter rows where 'AvgSurfT_tavg' is not equal to -9999.0
    # Convert 'AvgSurfT_tavg' column to numeric, handling errors by coercing to NaN
    # df['AvgSurfT_tavg'] = pd.to_numeric(df['AvgSurfT_tavg'], errors='coerce')

    # # Filter rows where 'AvgSurfT_tavg' is not equal to -9999.0
    # df_filtered = df[df['AvgSurfT_tavg'] != -9999.0]
    # df_filtered = df[df['AvgSurfT_tavg'] != "-9999.0"]
    df['AvgSurfT_tavg'] = pd.to_numeric(df['AvgSurfT_tavg'], errors='coerce')
    df_filtered = df.loc[(df.AvgSurfT_tavg != -9999.0)]
    # Now, df is your DataFrame with columns named according to the variable names
    # Save DataFrame to CSV
    file_name = file.split(".")[0]
    df_filtered.to_csv(f"../dataset/WLDAS_variables/{file_name}.csv", index=False)
    # pickle.dump(data_dict, open(f"../dataset/WLDAS_arrays/{file_name}.pkl", "wb"))

