import netCDF4 as nc
import os
from matplotlib import pyplot as plt

folder_dir = "../dataset/WLDAS"
for fid, file in enumerate(os.listdir(folder_dir)):
    if not file.endswith('.nc'):
        continue
    file_path = os.path.join(folder_dir, file)
    ds = nc.Dataset(file_path)

    # printing all variables and their shapes
    for v in ds.variables:
        print(v, ds.variables[v].shape)

    #rain_fall = ds.variables["Rainf_f_tavg"][:].astype(float) # retrieving the rainfall variable
    # plt.imsave(f"rain_{fid}.png", rain_fall[0,::-1], cmap='viridis')

    break
