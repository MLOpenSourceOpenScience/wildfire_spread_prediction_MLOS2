from pyhdf import SD
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

folder_path = 'dataset/daily_land_surface_temp_2017' # land surface temp data in 2017 for a region in the US SW(44.48443,-107.4375), NE(46.45309,-106.3125)
folder = os.listdir(folder_path)

for fidx, file_name in tqdm(enumerate(folder), total=len(folder)):
    if not file_name.endswith('.hdf'):
        continue
    file_path = os.path.join(folder_path, file_name)
    hdf_file = SD.SD(file_path)
    for d in hdf_file.datasets():
        print(d)
    # lst_data = hdf_file.select("LST_Day_1km")[:] 
    # for the original data, temperature is 50 times the actual value in Kelvin. Where there is no data, the value is 0.
    # no_data = lst_data==0
    # lst_data = lst_data * 0.02 - 273.15 # convert to celcius
    # lst_data[no_data] = 0

    # plt.imsave(f"temp_plots/{file_name}.png", lst_data, cmap='viridis')
    break
#hi