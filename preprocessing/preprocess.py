import netCDF4 as nc
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

print("Reading in fire data...")
gdf = gpd.read_file('dataframe/dataframe.shp')
gdf.dropna(inplace=True)
print("Finished reading in fire data")
discretized_lat_lon = False

folder_dir = "../dataset/WLDAS"
for fid, file in tqdm(enumerate(sorted(os.listdir(folder_dir))), total=len(os.listdir(folder_dir))):

    # actual start and end dates of fire are 2017-07-16 and 2017-08-24

    start_date = "WLDAS_NOAHMP001_DA1_20170709"
    end_date = "WLDAS_NOAHMP001_DA1_20170831"
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
        fire_data_path = "../dataset/fire_data_2017/fire_archive_SV-C2_425362.shp"
        gdf = gpd.read_file(fire_data_path)
        gdf["lat_bins"] = pd.cut(gdf["LATITUDE"], bins=lat, precision=10, labels=False)  # remove latitudes and longitudes that are not in western North America
        gdf["lon_bins"] = pd.cut(gdf["LONGITUDE"], bins=lon, precision=10, labels=False)
        gdf["lat_bins"] = len(lat) - gdf["lat_bins"] - 1
        gdf.dropna(inplace=True)
        discretized_lat_lon = True


    date = f"{file[20:24]}-{file[24:26]}-{file[26:28]}"
    fires = gdf[gdf["ACQ_DATE"].str.contains(date, na=False)]

    fire_array = np.zeros((len(lat), len(lon), 3), dtype=np.uint8)
    fire_array[fires["lat_bins"].astype(int), fires["lon_bins"].astype(int), 0] = 255
    
    plt.imsave(f"../dataset/Detwiler_Fire/{date}.png", fire_array)

    data_dict = {
        "time": time,
        "date": date,
        "lat": lat,
        "lon": lon,
        "data": data_array,
        "fire": fire_array
    }


    file_name = file.split(".")[0]
    pickle.dump(data_dict, open(f"../dataset/WLDAS_arrays/{file_name}.pkl", "wb"))

