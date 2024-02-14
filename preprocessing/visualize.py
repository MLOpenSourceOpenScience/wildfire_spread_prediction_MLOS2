import netCDF4 as nc
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

# import cv2
# import  imageio



folder_dir = "../dataset/WLDAS"
lat = None
lon = None
moisture_maps = []
for fid, file in tqdm(enumerate(sorted(os.listdir(folder_dir))), total=len(os.listdir(folder_dir))):
    if not file.endswith('.nc'):
        continue
    name_contains = "WLDAS_NOAHMP001_DA1_201712"
    if file[:len(name_contains)] != name_contains:
        continue
    if file[:len(name_contains)+2] == "WLDAS_NOAHMP001_DA1_20171211":
        moisture_maps.append(moisture_maps[-1])
        continue
    file_path = os.path.join(folder_dir, file)
    ds = nc.Dataset(file_path)
    lat = ds.variables["lat"][:].astype(float)
    lon = ds.variables["lon"][:].astype(float)
    moisture = ds.variables["SoilMoi00_10cm_tavg"][:].astype(float)
    moisture_maps.append(moisture[0, ::-1])

print(lat)
print(lat.shape)
print(lon)
print(lon.shape)


fire_data_path = "../dataset/fire_data_2017/fire_archive_SV-C2_425362.shp"
gdf = gpd.read_file(fire_data_path)
gdf["lat_bins"] = pd.cut(gdf["LATITUDE"], bins=lat, precision=10, labels=False)  # remove latitudes and longitudes that are not in western North America
gdf["lon_bins"] = pd.cut(gdf["LONGITUDE"], bins=lon, precision=10, labels=False)
gdf["lat_bins"] = len(lat) - gdf["lat_bins"] - 1
gdf.to_file('dataframe.shp') 
print("done")
gdf = gpd.read_file('dataframe.shp')

gdf = gdf[(gdf["lon_bins"]>=500) & (gdf["lon_bins"]<750)]
gdf = gdf[(gdf["lat_bins"]>=1800) & (gdf["lat_bins"]<1900)]
gdf["lon_bins"] -= 500
gdf["lat_bins"] -= 1800
gdf.dropna(inplace=True)

gdf.to_file('cropped_dataframe.shp') 
print("done")
gdf = gpd.read_file('cropped_dataframe.shp')

for idx, map in tqdm(enumerate(moisture_maps), total=len(moisture_maps)):
    # plt.imshow(map[1700:2000, 500:1000], cmap="Blues")
    plt.imshow(map[1800:1900, 500:750], cmap="Blues")
    plt.colorbar()
    plt.title(f"Thomas Fire December {idx+1}, 2017")
    fires = gdf[gdf["ACQ_DATE"].str.contains(f"2017-12-0{idx+1}", na=False)]
    plt.scatter(fires["lon_bins"], fires["lat_bins"], c="r", marker='o', s=5)
    plt.savefig(f"fire_maps_cropped/day_0{idx+1}.png")
    plt.close()
