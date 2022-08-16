from netCDF4 import Dataset
import numpy as np
import datetime as dt
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndimage


from netCDFfunc.utility import get_data_by_date, get_data_A, nc_write


grid_size = [0.01, 0.05, 0.10, 0.081, 0.054]
#grid_size = [0.05, 0.10, 0.081, 0.054]
base_dir = '/Volumes/T7/new_data/processed_data/processed_data_1_rok_avg'

for file in os.listdir(base_dir):
    
    if file != '30_years_dataset_1_rok_0220.nc': continue
    
    value1 = Dataset(f'{base_dir}/{file}', 'r', format='NETCDF4')
    
    value_1 = value1['avgsst'][:][0].data
    f_date = file[-7:-3]

    base_dir = '/Volumes/T7/intermediate_output/resize_test'

    for grid in grid_size :
        file_name = f'Resize_test_{f_date}_{grid}'
        
        nc_path = os.path.join(base_dir ,file_name+'.nc')
        #img_path = os.path.join(base_dir, f'img/resize_test_avgsst_2/{grid}' ,file_name+'.jpg')

        ds_new = Dataset(nc_path, 'w', format='NETCDF4')
        title = f'Global 30 years(1981~2011) SST mean data'
        comment = f'Average SST calculated 1981/9/1 ~ 2011/8/31, regridded to {grid}'
        ratio = 0.25 / grid
        
        lat_range = (int(440*ratio), int(572*ratio))
        lon_range = (int(440*ratio), int(600*ratio))
        lat_range = (lat_range[0]-lat_range[0],lat_range[1]-lat_range[0])
        lon_range = (lon_range[0]-lon_range[0],lon_range[1]-lon_range[0])

        data = ndimage.zoom(value_1, ratio, order=0) # nearest interpolation

        variable_name = 'avgsst'
        variable_standard_name = 'averageSST'
        variable_unit = 'degree C'
        variable_dtype = np.float32
        variable_values = data

        ds_new = nc_write(ds_new, title, comment, grid, variable_name, variable_standard_name, variable_unit, variable_dtype, variable_values, lat_range, lon_range)

        #save_img(data, img_path, figsize=(data.shape[1]/ratio, data.shape[0]/ratio))

        print("Grid: %.3f created" %grid)

        ds_new.close()
    print(f"{f_date} finished.")
