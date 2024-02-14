import netCDF4 as nc
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
import tqdm

folder_dir = "../dataset/WLDAS"
for fid, file in tqdm(enumerate(os.listdir(folder_dir))):
    if not file.endswith('.nc'):
        continue
    file_path = os.path.join(folder_dir, file)
    ds = nc.Dataset(file_path)

    # printing all variables and their shapes

    time = ds.variables["time"][:].astype(float) # retrieving the rainfall variable
    variables = {
        "Rainf_tavg",
        "TVeg_tavg",
        "Wind_f_tavg",
        "Tair_f_tavg",
        "Qair_f_tavg",
        "SoilMoi00_10cm_tavg",
        "SoilTemp00_10cm_tavg",
        "AvgSurfT_tavg",
    }

    data_list = []
    for var in variables:
        data = ds.variables[var][:].astype(float)
        data_list.append(data)
    data_array = np.array(data_list)

    data_dict = {
        "time": time,
        "data": data_array
    }

    file_name = file.split(".")[0]
    pickle.dump(data_dict, open(f"../dataset/WLDAS_arrays/{file_name}.pkl", "wb"))

    

    