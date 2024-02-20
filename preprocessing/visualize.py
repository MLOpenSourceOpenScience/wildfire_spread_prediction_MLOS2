import netCDF4 as nc
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

import cv2
import  imageio



folder_dir = "../dataset/WLDAS"
lat = None
lon = None
moisture_maps = []
for fid, file in tqdm(enumerate(sorted(os.listdir(folder_dir))), total=len(os.listdir(folder_dir))):
    if not file.endswith('.nc'):
        continue
    file_path = os.path.join(folder_dir, file)
    ds = nc.Dataset(file_path)
    lat = ds.variables["lat"][:].astype(float)
    lon = ds.variables["lon"][:].astype(float)
    moisture = ds.variables["SoilMoi00_10cm_tavg"][:].astype(float)
    moisture_maps.append(moisture[0, ::-1])


fire_data_path = "../dataset/fire_data_2017/fire_archive_SV-C2_425362.shp"
gdf = gpd.read_file(fire_data_path)
gdf["lat_bins"] = pd.cut(gdf["LATITUDE"], bins=lat, precision=10, labels=False)  # remove latitudes and longitudes that are not in western North America
gdf["lon_bins"] = pd.cut(gdf["LONGITUDE"], bins=lon, precision=10, labels=False)
gdf["lat_bins"] = len(lat) - gdf["lat_bins"] - 1
gdf.to_file('dataframe/dataframe.shp') 
print("done")
gdf = gpd.read_file('dataframe/dataframe.shp')
gdf.dropna(inplace=True)

gdf = gdf[(gdf["lon_bins"]>=500) & (gdf["lon_bins"]<750)]
gdf = gdf[(gdf["lat_bins"]>=1800) & (gdf["lat_bins"]<1900)]
gdf["lon_bins"] -= 500
gdf["lat_bins"] -= 1800
gdf.dropna(inplace=True)

gdf.to_file('cropped_dataframe.shp') 
print("done")
gdf = gpd.read_file("cropped_dataframe/cropped_dataframe.shp")
fires = gdf[gdf["ACQ_DATE"].str.contains(f"2017-07", na=False)]

for idx, map in tqdm(enumerate(moisture_maps), total=len(moisture_maps)):
    # plt.imshow(map[1700:2000, 500:1000], cmap="Blues")
    plt.imshow(map[1800:1900, 500:750], cmap="Blues")
    plt.colorbar()
    plt.title(f"Thomas Fire December {idx+1}, 2017")
    fires = gdf[gdf["ACQ_DATE"].str.contains(f"2017-12-0{idx+1}", na=False)]
    plt.scatter(fires["lon_bins"], fires["lat_bins"], c="r", marker='o', s=5)
    plt.savefig(f"fire_maps_cropped/day_0{idx+1}.png")
    plt.close()


map = plt.imread("map.png")
map *= 255
map = map.astype(np.uint8)
fire_dir = "../dataset/Detwiler_Fire"
files = sorted(os.listdir(fire_dir))
for f, fire in tqdm(enumerate(files), total=len(files)):
    if not fire.endswith(".png"):
        continue
    fire_path = os.path.join(fire_dir, fire)
    fire_map = plt.imread(fire_path)
    overlayed_map = map[:,:,:3].copy()
    # fire_map *= 255
    # print(fire_map[:,:,0])
    print(fire_map.shape)
    overlayed_map[fire_map==[255,0,0,0]] = [255,0,0]
    plt.imsave(f"../dataset/Detwiler_Fire_Overlayed_On_Map/{fire}", overlayed_map)
    # plt.close()
    exit()


#creating visualization for Detweiler Fire
lat_arr = np.load("lat.npy")
lon_arr = np.load("lon.npy")
lat = 37.61757
lon = -120.21321
# [37.61757,-120.21321]
lat_idx = len(lat_arr) - np.abs(lat_arr - lat).argmin()
lon_idx = np.abs(lon_arr - lon).argmin()


map = cv2.imread("map.png")
fire_dir = "../dataset/Detweiler_Fire"
files = sorted(os.listdir(fire_dir))
cropped_map_list = []
map_list = []
for f, fire in tqdm(enumerate(files), total=len(files)):
    if not fire.endswith(".png"):
        continue
    fire_path = os.path.join(fire_dir, fire)
    fire_map = cv2.imread(fire_path)
    overlayed_map = map.copy()
    overlayed_map[fire_map[:,:,2]==255] = [0,0,255]

    date = os.path.splitext(fire)[0]

    # Choose the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 5 
    font_color = (0, 0, 0)
    position = (200, 200) 
    
    cv2.putText(overlayed_map, date, position, font, font_scale, font_color, thickness=1, lineType=cv2.LINE_AA)
    map_list.append(overlayed_map[:,:,::-1].copy())

    font_scale = 0.5
    font_color = (0, 0, 0)
    position = (50, 20) 
    cropped_overlayed_map = cv2.putText(overlayed_map[lat_idx-100:lat_idx+100, lon_idx-100:lon_idx+100], date, position, font, font_scale, font_color, thickness=1, lineType=cv2.LINE_4)
    cropped_map_list.append(cropped_overlayed_map[:,:,::-1])

imageio.mimsave('../dataset/Detweiler_Fire_Overlayed_On_Map/Detwiler_Fire_Cropped.gif', cropped_map_list, fps=2)
imageio.mimsave('../dataset/Detweiler_Fire_Overlayed_On_Map/Detwiler_Fire.gif', map_list, fps=2)