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

    
    if ds_name == 'CMC_SST' :
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
        
    if ds_name == 'NAVO_SST':
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

    elif ds_name == 'CMC_SST' or ds_name == 'GAMSSA_SST':
        np.place(ice, ice[:,:] != 8, False)
        np.place(ice, ice[:,:] == 8, True)
        
    elif ds_name == 'DMI_SST' or ds_name == 'MUR_SST' or ds_name == 'MUR' or ds_name == 'MW_IR_OI_SST' or ds_name == 'OSTIA_SST' or ds_name == 'MW_OI_SST':
        np.place(ice, ice[:,:] != 9, False)
        np.place(ice, ice[:,:] == 9, True)
        
    elif ds_name == 'OSPO_SST' : #or ds_name == 'OSPO_SST_Night':
        np.place(ice, ice[:,:] != 4, False)
        np.place(ice, ice[:,:] == 4, True)
        
    elif ds_name == 'NAVO_SST' :
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


def to_nc(nc_file_path, ds_name, data, avg_data, period, region, grid, date, is_grade=False):
    
    variable_unit = 'degree C'
    variable_dtype = np.float32
    
    if is_grade == False :
        variable_name = 'YY30_AVG_SST_ANOMALY'
        variable_standard_name = 'SST Anomaly'
            
    elif is_grade == True :
        variable_name = 'SST_GRD_CD'
        variable_standard_name = 'SST Grade'

    
    ds_new = Dataset(nc_file_path, 'w', format='NETCDF4')

    if period == 1 :
        yy_start_date = '1981.09.01'
        yy_end_date = '2011.08.31'
    elif period == 2 :
        yy_start_date = '1991.01.01'
        yy_end_date = '2020.12.31'

    # if is_grade == True : 
    #     data_type = 'grade'
    # elif is_grade == False : 
    #     data_type = 'anomaly'
    
    # title = f'{region} 30 years({year_range}) base SST {data_type} data ({date})' 
    # comment = f'SST based on {date_range}'
    
    avg_variable_name = 'YY30_AVG_SST'
    avg_variable_standard_name = '30years SST Average'
    avg_variable_values = avg_data
    
    variable_values = data
    ratio = 0.25 / grid
    
    lat_range = (round(440*ratio), round(572*ratio))
    lon_range = (round(440*ratio), round(600*ratio))

    ds_new = nc_write(ds_new, grid, ds_name, yy_start_date, yy_end_date, region, date,
                      variable_name, variable_standard_name, variable_values, 
                      avg_variable_name, avg_variable_standard_name, avg_variable_values,
                      variable_unit, variable_dtype,  
                      lat_range, lon_range,
                      not_use_avg = is_grade)
    
    ds_new.close()
    
    
def to_csv(nc_file_path, csv_file_path, include_null=False) :
    
    d_stack = []
    ds = Dataset(nc_file_path, 'r', format='NETCDF4')
    
    if 'anomaly' in nc_file_path :
        anomaly = ds['YY30_AVG_SST_ANOMALY'][:].data
        avg = ds['YY30_AVG_SST'][:].data
    elif 'grade' in nc_file_path :
        grade = ds['SST_GRD_CD'][:].data
    lat_range = ds['LA'][:]
    lon_range = ds['LO'][:]
    
    if 'anomaly' in nc_file_path :
        for i in range(len(lat_range)) :
            for j in range(len(lon_range)):
                if include_null == False and (anomaly[i][j] == -32768 or anomaly[i][j] == 32767 or np.isnan(anomaly[i][j])):
                    continue
                else :
                    d_stack.append((round(lat_range[i],3), round(lon_range[j],3), anomaly[i][j], avg[i][j]))
                    
        df = pd.DataFrame(d_stack, columns=['LA', 'LO', 'YY30_AVG_SST_ANOMALY', 'YY30_AVG_SST'])
    
    if 'grade' in nc_file_path :
        for i in range(len(lat_range)) :
            for j in range(len(lon_range)):
                if include_null == False and (grade[i][j] == -32768  or grade[i][j] == 32767 or np.isnan(grade[i][j])):
                    continue
                else :
                    d_stack.append((round(lat_range[i],3), round(lon_range[j],3), grade[i][j]))
        df = pd.DataFrame(d_stack, columns=['LA', 'LO', 'SST_GRD_CD'])
        
    df.to_csv(csv_file_path, index=True)
    
    ds.close()