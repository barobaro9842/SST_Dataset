from netCDF4 import Dataset
from netCDFfunc.utility import *

import numpy as np

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
        
    if ds_name == 'DMI' :
        sst = np.roll(sst, 18, axis=0)
        ice = np.roll(ice, 18, axis=0)
        
    if ds_name == 'NAVO':
        sst = np.flip(sst, axis=0)
        ice = np.flip(ice, axis=0)
        
        sst = np.roll(sst, -1, axis=0)
        ice = np.roll(ice, -1, axis=0)
        
        sst = np.roll(sst, -1, axis=1)
        ice = np.roll(ice, -1, axis=1)
    
    np.place(sst, sst[:,:] <= -32767., 32767)
    sst = sst - 273.15
    
    # ice
    if ds_name == 'AVHRR' :
        ice = data_dic['ice'].copy()
        np.place(ice, ice[:,:] != -128, True)
        np.place(ice, ice[:,:] == -128, False)

    elif ds_name == 'CMC' or ds_name == 'GAMSSA':
        np.place(ice, ice[:,:] != 8, False)
        np.place(ice, ice[:,:] == 8, True)
        
    elif ds_name == 'DMI' or ds_name == 'MUR25' or ds_name == 'MUR' or ds_name == 'MW' or ds_name == 'OSTIA' or ds_name == 'MWIR':
        np.place(ice, ice[:,:] != 9, False)
        np.place(ice, ice[:,:] == 9, True)
        
    elif ds_name == 'OSPO' or ds_name == 'OSPON':
        np.place(ice, ice[:,:] != 4, False)
        np.place(ice, ice[:,:] == 4, True)
        
    elif ds_name == 'NAVO' :
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