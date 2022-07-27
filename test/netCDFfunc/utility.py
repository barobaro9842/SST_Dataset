from .preprocessing import get_data_A
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from netCDF4 import Dataset

import datetime as dt
import os


def check_err_mask_A(variable_name) -> (int, list):
    months = range(1,13)
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    err_date = []
    cnt = 0
    first = True
    
    for year in tqdm(range(1981,2022)):
        for month, day_len in zip(months, days):
            for date in range(1,day_len+1):
                if year == 1981 and month in [1,2,3,4,5,6,7,8]:
                    continue
                if year == 2021 and month == 1 and date == 16 :
                    continue
                    
                if first == True :
                    basis = get_data_A(year,month,date, variable_name, is_mask=True)
                    first = False

                else :
                    check = basis == get_data_A(year,month,date, variable_name, is_mask=True)
                    cnt += (check == False).sum()
                    if False in check :
                        err_date.append((year, month, date))
                
    return cnt, err_date
                        
def show_img(arr):
    fig = plt.figure(figsize=(30,20))
    plt.imshow(arr)
    plt.show()
    
    
def to_video(arr,frame,output_path):
    '''
    arr = [time, lat, lon]
    '''
    length = arr.shape[0]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame, arr.shape[2], arr.shape[1], False)
    
    for i in range(length) :
        data = arr[i]
        out.write(data)
        
    out.release()
    
def get_data_sequence(get_data_func, var_name ,start_date : tuple, end_date : tuple, is_mask=False) -> list:
    
    s_year, s_month, s_day = start_date
    e_year, e_month, e_day = end_date
    
    
    result = []
    
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    for year in tqdm(range(s_year, e_year+1)) :
        
        if year == s_year : months = range(s_month, 13)
        elif year == e_year : months = range(1, e_month+1)
        else : months = range(1,13)
        
        for month, day_len in zip(months, days[months[0]-1:]):
            
            if year == s_year and month == s_month : day_range = range(s_day, day_len+1)
            elif year == e_year and month == e_month : day_range = range(1, e_day+1)
            else : day_range = range(1, day_len+1)
            
            for day in day_range :
                if get_data_func.__name__ == 'get_data_A':
                    if year == 2021 and month == 1 and day == 16 : continue

                result.append(get_data_func(year,month,day, var_name, is_mask=is_mask))
        
    return result
    
    
def get_data_by_date(get_data_func, var_name ,start_date : tuple, end_date : tuple, is_mask=False) -> list:
    s_year, s_month, s_day = start_date
    e_year, e_month, e_day = end_date
    
    result_dic = dict()
    result = []
    
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    for year in tqdm(range(s_year, e_year+1)) :
        
        if year == s_year : months = range(s_month, 13)
        elif year == e_year : months = range(1, e_month+1)
        else : months = range(1,13)
        
        for month, day_len in zip(months, days[months[0]-1:]):
            
            if year == s_year and month == s_month : day_range = range(s_day, day_len+1)
            elif year == e_year and month == e_month : day_range = range(1, e_day+1)
            else : day_range = range(1, day_len+1)
            
            for day in day_range :
                if get_data_func.__name__ == 'get_data_A':
                    if year == 2021 and month == 1 and day == 16 : continue
            
                data = get_data_func(year,month,day, var_name, is_mask=is_mask)
                if result_dic.get((month, day)) : result_dic[(month, day)].append(data)
                else : result_dic[(month, day)] = [data]
                
        
    return result_dic

def get_stat(input, type) -> dict:
    
    '''
    type : mean, cnt, perc
    '''
    result_dic = dict()

    days = [31,28,31,30,31,30,31,31,30,31,30,31]

    if type == 'mean':
        for month, day_len in tqdm(zip(range(1,13), days)):
            for day in range(1,day_len+1):
                mean_arr = np.mean(np.array(input[(month, day)]), axis=0)
                result_dic[(month, day)] = mean_arr.tolist()
                
    if type == 'cnt' :
        for month, day_len in tqdm(zip(range(1,13), days)):
            for day in range(1,day_len+1):
                result_dic[(month, day)] = np.count_nonzero(np.array(input[(month, day)]) != -999, axis=0).tolist()
                
    
    if type == 'perc' :
        for month, day_len in tqdm(zip(range(1,13), days)):
            for day in range(1,day_len+1):
                max_arr = np.max(np.array(input[(month, day)]), axis=0)
                np.place(max_arr, max_arr[:,:] == -999, np.nan)
                result_dic[(month, day)] = (max_arr * 0.9).tolist()
                max_arr.fillna(-999)
                
    
    return result_dic


def create_new_variable(dsin, dsout, new_varable_name, dtype, dimension, fill_value, values, attributes):
    '''
    ex)
    variable_name = 'mean_sst'
    dtype = np.float32
    dimensions = ('time', 'zlev', 'lat', 'lon')
    fill_value = -999
    values = mean[(9,1)]
    attribute = {'long_name' : 'mean sst for 30 years (1981/9/1~2011/8/31)',
                'add_offset' : 0.0,
                'scale_factor' : 0.01,
                'valid_min' : -1200,
                'valid_max' : 1200,
                'units' : 'celcius'}
    '''
    
    
    for attr in dsin.ncattrs() :
        dsout.setncattr(attr, dsin.getncattr(attr))

    for k, v in dsin.dimensions.items():
        dsout.createDimension(k, v.size)
    
    for name, variable in dsin.variables.items():
    
        existing_variable = dsout.createVariable(name, variable.datatype, variable.dimensions, fill_value=fill_value) # name, datatype, dimension
        dsout[name][:] = dsin[name][:] # values

        for attr in variable.ncattrs():
            if attr == '_FillValue': continue
            existing_variable.setncattr(attr, variable.getncattr(attr)) # variable attr
            
    new_variable = dsout.createVariable(new_varable_name, dtype, dimension, fill_value=fill_value)
    dsout[new_varable_name][:] = values
    
    for k,v in attributes.items():
        if attr == '_FillValue': continue
        new_variable.setncattr(k, v) # variable attr
    
    return dsout


def transfer_data(year, month, day, variable_type, variable_value): 
    '''
    variable_type = 'mean' or 'perc' or 'cnt'
    '''
    
    date = dt.date(year,month,day).strftime('%Y%m%d')
    
    if year < 2016 : 
        dsin = Dataset(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/oisst-avhrr-v02r01.{date}.nc', 'r', format='NETCDF4')
    else :
        if os.path.exists(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'):
            dsin = Dataset(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc', 'r', format='NETCDF4')
        else :
            dsin = Dataset(f'/Volumes/T7/AVHRR_OI_SST/v2.1/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.nc', 'r', format='NETCDF4')
    
    if not os.path.exists(f'/Volumes/T7/new_data/AVHRR_OI_SST/{year}'):
        os.makedirs((f'/Volumes/T7/new_data/AVHRR_OI_SST/{year}'))
                           
    dsout = Dataset(f'/Volumes/T7/new_data/AVHRR_OI_SST/{year}/{date}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.nc', 'w', format='NETCDF4')

    if variable_type == 'mean' :
        variable_name = 'mean_sst'
        dtype = np.float32
        attribute = {'long_name' : 'mean sst for 30 years (1981/9/1~2011/8/31)',
                     'add_offset' : 0.0,
                     'scale_factor' : 0.01,
                     'valid_min' : -1200,
                     'valid_max' : 1200,
                     'units' : 'celcius'}
        
    if variable_type == 'perc' :
        variable_name = '90-percentile_sst'
        dtype = np.float32
        attribute = {'long_name' : '90-percentile of sst for 30years (1981/9/1~2011/8/31)',
                     'add_offset' : 0.0,
                     'scale_factor' : 0.01,
                     'valid_min' : -1200,
                     'valid_max' : 1200,
                     'units' : 'celcius'}
    
    if variable_type == 'cnt' :
        variable_name = 'cnt_of_value'
        dtype = np.int16
        attribute = {'long_name' : 'based values counted by pixel (30,0 = normal, other = anormal)',
                     'add_offset' : 0,
                     'scale_factor' : 1,
                     'valid_min' : 0,
                     'valid_max' : 30,
                     'units' : '-'}
        
              
    
    # common attribute
    dimensions = ('time', 'zlev', 'lat', 'lon')
    fill_value = -999
    values = variable_value[(month,date)]

    create_new_variable(dsin, dsout, variable_name, dtype, dimensions, fill_value, values, attribute)

    dsout.close()
    dsin.close()