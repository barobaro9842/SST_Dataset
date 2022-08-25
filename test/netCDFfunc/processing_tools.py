from netCDF4 import Dataset
from netCDFfunc.utility import *

import numpy as np

import os

import shutil
import requests
import datetime

def download_data(download_path, start_date=None, end_date=None, dataset_names=None):
    ''' 
    start_date and end_date = Tuple (Year, Month, Day)
    
     1) start_date and end_date == None : most_recent
     2) start_date != None and end_date == None : start_date ~ most_recent
     3) start_date != None and end_date != None : start_date ~ end_date

    dataset_names = list of (AVHRR, CMC, DMI, GAMSSA, MUR25, MUR0.01, MWIR, MW, NAVO, OSPON, OSPO, OSTIA)
    '''
    
    common_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/ghrsst/data'

    if dataset_names == None :
        dataset_names = ['AVHRR', 'CMC', 'DMI', 'GAMSSA', 'MUR25', 'MUR', 'MWIR', 'MW', 'NAVO', 'OSPON', 'OSPO', 'OSTIA']

    if start_date == None and end_date == None : 
        date_range = range(6,0,-1)
        end_date = datetime.datetime.now()
    else: 
        start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        date_range = range((end_date - start_date).days, -1, -1)
        
    for delta in date_range:
        
        recent = end_date - datetime.timedelta(days=delta)
        year = recent.strftime('%Y')
        date = recent.strftime('%Y%m%d')
        
        j_day = recent.strftime('%j')
        j_day = '%03d' % int(j_day)

        for dataset_name in dataset_names :
            
            if dataset_name == 'AVHRR':
                file_name = f'{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'
                url = f'{common_url}/GDS2/L4/GLOB/NCEI/AVHRR_OI/v2.1/{year}/{j_day}/{file_name}'
            
            elif dataset_name == 'CMC' :
                file_name = f'{date}120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/CMC/CMC0.1deg/v3/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'DMI' :
                file_name = f'{date}000000-DMI-L4_GHRSST-SSTfnd-DMI_OI-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/DMI/DMI_OI/v1/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'GAMSSA' :
                file_name = f'{date}120000-ABOM-L4_GHRSST-SSTfnd-GAMSSA_28km-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/ABOM/GAMSSA/v1.0/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'MUR25' :
                file_name = f'{date}090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc'
                url = f'{common_url}/GDS2/L4/GLOB/JPL/MUR25/v4.2/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'MUR' :
                file_name = f'{date}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
                url = f'{common_url}/GDS2/L4/GLOB/JPL/MUR/v4.1/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'MWIR' :
                file_name = f'{date}120000-REMSS-L4_GHRSST-SSTfnd-MW_IR_OI-GLOB-v02.0-fv05.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/REMSS/mw_ir_OI/v5.0/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'MW' :
                file_name = f'{date}120000-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/REMSS/mw_OI/v5.0/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'NAVO' :
                file_name = f'{date}000000-NAVO-L4_GHRSST-SST1m-K10_SST-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/NAVO/K10_SST/v1/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'OSPON' :
                file_name = f'{date}000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended_Night-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/OSPO/Geo_Polar_Blended_Night/v1/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'OSPO' :
                file_name = f'{date}000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/OSPO/Geo_Polar_Blended/v1/{year}/{j_day}/{file_name}'
                
            elif dataset_name == 'OSTIA' :
                file_name = f'{date}-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIA.nc.bz2'
                url = f'{common_url}/L4/GLOB/UKMO/OSTIA/{year}/{j_day}/{file_name}'

            if file_name.endswith('.bz2') : file_name = file_name.replace('.bz2', '')
            target_file = os.path.join(download_path, dataset_name, file_name)
            
            if os.path.exists(target_file) and os.path.getsize(target_file) > 5 * 1024:
                if dataset_name == 'MUR' :
                    if os.path.getsize(target_file) > 600 * 1024 * 1024:    
                        print(f'[{date}|{dataset_name}] is already downloaded!')
                        continue
                else : 
                    print(f'[{date}|{dataset_name}] is already downloaded!')
                    continue
            
            response = requests.get(url, stream=True)
            
            if response.status_code == 404:
                print(f'[{date}|{dataset_name}] doesn\'t exist!')
                continue
            
            if not os.path.exists(os.path.join(download_path, dataset_name)) : os.mkdir(os.path.join(download_path, dataset_name))
            
            with open(target_file, 'wb') as out_file:
                print(f'[{date}|{dataset_name}] downloading...')
                shutil.copyfileobj(response.raw, out_file)
                
            del response
            
            

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