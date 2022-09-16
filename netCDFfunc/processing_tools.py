from netCDF4 import Dataset
from netCDFfunc.utility import *

import numpy as np
import pandas as pd

import os

import shutil
import requests
import datetime

def get_date(file_name):
    return file_name[:8]


def get_meta_data(raw_data_path, date, file_name):
        
    data_file = os.path.join(raw_data_path, file_name)
    ds_in = Dataset(data_file, 'r', format='NETCDF4')
    
    sst = ds_in.variables['analysed_sst'][:].data[0]
    mask = ds_in.variables['mask'][:].data[0]
    ice = ds_in.variables['sea_ice_fraction'][:].data[0]
    
    try :
        if 'MW_IR' in file_name :
            grid = 0.08789
        else :
            grid = np.float32(ds_in.geospatial_lat_resolution)
    except :
        grid = 0.05
        
    lat = ds_in.variables['lat'][:].data
    lon = ds_in.variables['lon'][:].data
    
    ds_in.close()
    
    return {'sst':sst, 'mask':mask, 'ice':ice, 'lat':lat, 'lon':lon, 'grid':grid}

def preprocessing_dataset(ds_name, data_dic) :
    
    sst = data_dic['sst'].copy()
    ice = data_dic['mask'].copy()

    
    if ds_name == 'CMC' :
        cmc_land = data_dic['mask'].copy()
        np.place(cmc_land, cmc_land[:,:] != 2, False)
        np.place(cmc_land, cmc_land[:,:] == 2, True)
        sst = masking(sst, cmc_land, fill_value=-32767)
        
        sst = np.roll(sst, -1, axis=0)
        ice = np.roll(ice, -1, axis=0)
        
        sst = np.roll(sst, -1, axis=1)
        ice = np.roll(ice, -1, axis=1)
        
        
    if ds_name == 'DMI_SST' :
        sst = np.roll(sst, 18, axis=0)
        ice = np.roll(ice, 18, axis=0)
        
    if ds_name == 'NAVO_K10_SST_GDS2':
        sst = np.flip(sst, axis=0)
        ice = np.flip(ice, axis=0)
        
        sst = np.roll(sst, -1, axis=0)
        ice = np.roll(ice, -1, axis=0)
        
        sst = np.roll(sst, -1, axis=1)
        ice = np.roll(ice, -1, axis=1)
    
    np.place(sst, sst[:,:] <= -32767., 32767)
    sst = sst - 273.15
    
    # ice
    if ds_name == 'AVHRR_OI_SST' :
        ice = data_dic['ice'].copy()
        np.place(ice, ice[:,:] != -128, True)
        np.place(ice, ice[:,:] == -128, False)

    elif ds_name == 'CMC' or ds_name == 'GAMSSA_GDS2':
        np.place(ice, ice[:,:] != 8, False)
        np.place(ice, ice[:,:] == 8, True)
        
    elif ds_name == 'DMI_SST' or ds_name == 'MUR_SST' or ds_name == 'MUR' or ds_name == 'MW_IR_SST' or ds_name == 'OSTIA_SST' or ds_name == 'MW_OI_SST':
        np.place(ice, ice[:,:] != 9, False)
        np.place(ice, ice[:,:] == 9, True)
        
    elif ds_name == 'OSPO_SST' or ds_name == 'OSPO_SST_Night':
        np.place(ice, ice[:,:] != 4, False)
        np.place(ice, ice[:,:] == 4, True)
        
    elif ds_name == 'NAVO_K10_SST_GDS2' :
        np.place(ice, ice[:,:], False)
        
    return sst, ice


def get_path(base_path, date):
    
    anomaly_path = os.path.join(base_path, 'anomaly')
    if not os.path.exists(anomaly_path) : os.mkdir(anomaly_path)
        
    grade_path = os.path.join(base_path, 'grade')
    if not os.path.exists(grade_path) : os.mkdir(grade_path)
    
    anomaly_path = os.path.join(anomaly_path, f'{date[:4]}')
    if not os.path.exists(anomaly_path) : os.mkdir(anomaly_path)
    
    grade_path = os.path.join(grade_path, f'{date[:4]}')
    if not os.path.exists(grade_path) : os.mkdir(grade_path)
    
    return anomaly_path, grade_path


def to_nc(nc_file_path, data, period, region, grid, date, is_grade=False):
    
    if is_grade == False :
        variable_name = 'anomalysst'
        variable_standard_name = 'SST Anomaly'
        variable_unit = 'degree C'
        variable_dtype = np.float32
    
    elif is_grade == True :
        variable_name = 'grade'
        variable_standard_name = 'abnormal sst grade'
        variable_unit = 'degree C'
        variable_dtype = np.float32
    
    ds_new = Dataset(nc_file_path, 'w', format='NETCDF4')

    if period == 1 :
        year_range = '1981~2011'
        date_range = '1981/9/1~2011/8/31'
    elif period == 2 :
        year_range = '1991~2020'
        date_range = '1991/1/1~2020/12/31'

    if is_grade == True : 
        data_type = 'grade'
    elif is_grade == False : 
        data_type = 'anomaly'
    
    title = f'{region} 30 years({year_range}) base SST {data_type} data ({date})' 
    comment = f'SST based on {date_range}'
    variable_values = data
    ratio = 0.25 / grid
    
    lat_range = (round(440*ratio), round(572*ratio))
    lon_range = (round(440*ratio), round(600*ratio))

    ds_new = nc_write(ds_new, title, comment, grid, 
                      variable_name, variable_standard_name, variable_unit, variable_dtype, variable_values, 
                      lat_range, lon_range)
    ds_new.close()
    
    
def to_csv(nc_file_path, csv_file_path, include_null=False) :
    
    d_stack = []
    ds = Dataset(nc_file_path, 'r', format='NETCDF4')
    
    if 'anomaly' in nc_file_path :
        data = ds['anomalysst'][:].data
    elif 'grade' in nc_file_path :
        data = ds['grade'][:].data
    lat_range = ds['lat'][:]
    lon_range = ds['lon'][:]
    
    for i in range(len(lat_range)) :
        for j in range(len(lat_range)):
            if 'anomaly' in nc_file_path and include_null == False and (data[0][i][j] == -999 or data[0][i][j] == 32767):
                continue
            if 'grade' in nc_file_path and data[0][i][j] == 10 :
                continue
            
            d_stack.append((round(lat_range[i],4), round(lon_range[j],4), data[0][i][j]))
            
    df = pd.DataFrame(d_stack, columns=['lat', 'lon', 'sst'])
    df.to_csv(csv_file_path, index=False)
    
    ds.close()