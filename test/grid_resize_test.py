from netCDF4 import Dataset
import numpy as np
import datetime as dt
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndimage


from netCDFfunc.utility import get_data_by_date, get_data_A, to_img, nc_write


grid_size = [0.01, 0.05, 0.10, 0.081, 0.054, 0.25]
#grid_size = [0.05, 0.10, 0.081, 0.054]
base_dir = '/Volumes/T7/new_data/processed_data/processed_data_2_rok_ice' ################

for file in os.listdir(base_dir):
    
    ds = Dataset(f'{base_dir}/{file}', 'r', format='NETCDF4')
    
    value_1 = ds['ice'][:].data[0] ############
    f_date = file[-7:-3]

    output_dir = '/Volumes/T7/base_data/2/ice' #################

    for grid in grid_size :
        file_name = f'ice_2_rok_{f_date}_{grid}' ################
        
        nc_path = os.path.join(output_dir, f'{grid}' ,file_name+'.nc')
        #img_path = os.path.join(base_dir, f'img/resize_test_avgsst_2/{grid}' ,file_name+'.jpg')

        ds_new = Dataset(nc_path, 'w', format='NETCDF4')
        # title = f'Global 30 years(1991~2020) SST mean data' #####################
        # comment = f'Average SST calculated 1991/1/1 ~ 2020/12/31, regridded to {grid}'##############
        # title = f'Global 30 years(1991~2020) 90-percentile data'
        # comment = f'90-percentile SST calculated 1991/1/1 ~ 2020/12/31, regridded to {grid}'
        title = f'Global 30 years(1991~2020) Sea-ice data'
        comment = f'Calculated 1991/1/1 ~ 2020/12/31, pixels more than 15% ice as ice, less than 15% ice as not ice, regridded to {grid}'
        ratio = 0.25 / grid
        
        lat_range = (round(440*ratio), round(572*ratio))
        lon_range = (round(440*ratio), round(600*ratio))
        # lat_range = (lat_range[0]-lat_range[0],lat_range[1]-lat_range[0])
        # lon_range = (lon_range[0]-lon_range[0],lon_range[1]-lon_range[0])

        data = ndimage.zoom(value_1, ratio, order=0) # nearest interpolation

        variable_name = 'ice' ##################
        variable_standard_name = 'Sea Ice'  ##################
        variable_unit = '' ##################
        variable_dtype = np.int8 ##################
        variable_values = data

        ds_new = nc_write(ds_new, title, comment, grid, variable_name, variable_standard_name, variable_unit, variable_dtype, variable_values, lat_range, lon_range)

        ds_new.close()

        #save_img(data, img_path, figsize=(data.shape[1]/ratio, data.shape[0]/ratio))

        print("Grid: %.3f created" %grid)

    print(f"{f_date} finished.")
