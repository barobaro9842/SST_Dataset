from netCDF4 import Dataset
import numpy as np
import datetime as dt
import os
from tqdm.notebook import tqdm

def get_data_A(year, month, day, variable_name, is_mask=False) -> np.ndarray:

    '''
    varialbes = ['err', 'ice', 'lat', 'lon', 'sst', 'anom']
    '''
    
    date = dt.date(year,month,day).strftime('%Y%m%d')
    
    is_value = False
    
    if variable_name in ['anom', 'err', 'ice', 'sst'] :
        is_value = True
    
    if year < 2016 : 
        D = Dataset(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/oisst-avhrr-v02r01.{date}.nc', 'r', format='NETCDF4')

    else :
        if os.path.exists(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'):
            D = Dataset(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc', 'r', format='NETCDF4')
        else :
            D = Dataset(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.nc', 'r', format='NETCDF4')
        
        if variable_name == 'err' : variable_name = 'analysis_error'
        elif variable_name == 'sst' : variable_name = 'analysed_sst'
        elif variable_name == 'ice' : variable_name = 'sea_ice_fraction'
    
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
            
            data = data - 273.15 #kelvin to celcius
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
    

