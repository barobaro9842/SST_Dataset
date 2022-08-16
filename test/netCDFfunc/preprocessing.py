from netCDF4 import Dataset
import numpy as np
import datetime as dt
import os
from tqdm.notebook import tqdm

import shutil
import requests
import datetime


def download_data(output_path, start_date=None, end_date=None, dataset_names=None):
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

    if start_date == None and end_date == None : date_range = range(6,0,-1)
        
    for delta in date_range:
        
        recent = datetime.datetime.now()-datetime.timedelta(days=delta)
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
            target_file = os.path.join(output_path, dataset_name, file_name)
            
            if os.path.exists(target_file) and os.path.getsize(target_file) > 5 * 1024:
                print(f'[{date}|{dataset_name}] is already downloaded!')
                continue
            
            response = requests.get(url, stream=True)
            
            if response.status_code == 404:
                print(f'[{date}|{dataset_name}] doesn\'t exist!')
                continue
            
            if not os.path.exists(os.path.join(output_path, dataset_name)) : os.mkdir(os.path.join(output_path, dataset_name))
            
            with open(target_file, 'wb') as out_file:
                print(f'[{date}|{dataset_name}] downloading...')
                shutil.copyfileobj(response.raw, out_file)
                
            del response


def get_data_A(base_dir, year, month, day, variable_name, is_mask=False) -> np.ndarray:

    '''
    varialbes = ['err', 'ice', 'lat', 'lon', 'sst', 'anom']
    '''
    
    date = dt.date(year,month,day).strftime('%Y%m%d')
    
    is_value = False
    
    if variable_name in ['anom', 'err', 'ice', 'sst'] :
        is_value = True
    
    
    if year < 2016 :
        directory = os.path.join(base_dir, f'{year}/oisst-avhrr-v02r01.{date}.nc') 

    else :
        directory = os.path.join(base_dir, f'{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc')
        if not os.path.exists(directory):
            directory = os.path.join(base_dir, f'{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.nc')

        if variable_name == 'err' : variable_name = 'analysis_error'
        elif variable_name == 'sst' : variable_name = 'analysed_sst'
        elif variable_name == 'ice' : variable_name = 'sea_ice_fraction'
    
    D = Dataset(directory, 'r', format='NETCDF4')
    
    var = D[variable_name]
    masked_array = var[:]
    
    if year < 2016 : 
        if is_value == True :        
            data = masked_array.data[0][0]
            mask = masked_array.mask[0][0]

        else :
            data = masked_array.data
            mask = masked_array.mask
            
    else :
        if is_value == True :
            
            # 경도 기준점 이동 (180도, 720pixel)
            data = np.roll(masked_array.data[0], 720)
            mask = np.roll(masked_array.mask[0], 720)
            
            # 720 pixel 앞쪽으로 위도 1pixel씩 밀림
            data[:,:720] = np.roll(data[:,:720], -1, axis=0)
            mask[:,:720] = np.roll(mask[:,:720], -1, axis=0)
            
            data = data - 273.15 #wkelvin to celcius
            np.place(data, data[:,:]== -33041.15, -999)
            
        elif variable_name == 'lon' :
            # 2016년 이후 : 
            data = masked_array.data + 180
            mask = np.roll(masked_array.mask, 720)
        else :
            data = masked_array.data
            mask = masked_array.mask
            

    D.close()
    
    if is_mask == False : return data
    else : return mask
    

