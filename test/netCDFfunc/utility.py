from .preprocessing import get_data_A
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
import numpy as np
from netCDF4 import Dataset

import datetime as dt
import os


def check_err_mask_A(variable_name):
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
    fig = plt.figure(figsize=(72,36))
    np.place(arr, arr[:,:]==-999, np.nan)
    cmap = cm.jet.copy()
    cmap.set_bad(color='gray')
    plt.imshow(arr, cmap=cmap)
    plt.axis('off')
    #plt.savefig('test.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
def save_img(arr, output_path, figsize=()):
    if figsize == ():
        x, y = arr.shape
    else :
        x,y = figsize
        
    x = x/20
    y = y/20
    
    fig = plt.figure(figsize=(y, x))
    np.place(arr, arr[:,:]==-999, np.nan)
    cmap = cm.jet.copy()
    cmap.set_bad(color='gray')
    plt.imshow(arr, cmap=cmap)
    plt.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    
def get_data_sequence(base_dir, get_data_func, var_name ,start_date : tuple, end_date : tuple, is_mask=False) -> list:
    
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

                result.append(get_data_func(base_dir, year,month,day, var_name, is_mask=is_mask))
        
    return result
    
    
def get_data_by_date(base_dir, get_data_func, var_name ,start_date : tuple, end_date : tuple, specific_year=None, specific_date=(), is_mask=False) -> list:
    s_year, s_month, s_day = start_date
    e_year, e_month, e_day = end_date
    
    result_dic = dict()
    if specific_date != ():
        specific_month, specific_day = specific_date
    
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    for year in tqdm(range(s_year, e_year+1)) :
        
        if specific_year != None and year != specific_year : continue
        
        if year == s_year : months = range(s_month, 13)
        elif year == e_year : months = range(1, e_month+1)
        else : months = range(1,13)
        
        for month, day_len in zip(months, days[months[0]-1:]):
            
            if month != specific_month : continue
            
            if year == s_year and month == s_month : day_range = range(s_day, day_len+1)
            elif year == e_year and month == e_month : day_range = range(1, e_day+1)
            else : day_range = range(1, day_len+1)
            
            for day in day_range :
                
                if day != specific_day : continue
                
                if get_data_func.__name__ == 'get_data_A':
                    if year == 2021 and month == 1 and day == 16 : continue
            
                data = get_data_func(base_dir, year,month,day, var_name, is_mask=is_mask)
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
                np.place(max_arr, max_arr[:,:] == np.nan, -999)
                
    
    return result_dic


def copy_existing_variable(dsin, dsout, default_fill_value):
    
    for attr in dsin.ncattrs() :
        
        if attr == 'anom' : continue
        elif attr == 'err' : new_attr = 'analysis_error'
        elif attr == 'ice' : new_attr = 'sea_ice_fraction'
        elif attr == 'sst' : new_attr = 'analysed_sst'
        else : new_attr = attr
        
        dsout.setncattr(new_attr, dsin.getncattr(attr))

    for k, v in dsin.dimensions.items():
        dsout.createDimension(k, v.size)
    
    for name, variable in dsin.variables.items():
    
        existing_variable = dsout.createVariable(name, variable.datatype, variable.dimensions, fill_value=default_fill_value) # name, datatype, dimension
        dsout[name][:] = dsin[name][:] # values

        for attr in variable.ncattrs():
            if attr == '_FillValue': continue
            existing_variable.setncattr(attr, variable.getncattr(attr)) # variable attr
                
    return dsout


def create_new_variable(dsout, new_variable_name, dtype, dimension, fill_value, values, attributes):
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
            
    new_variable = dsout.createVariable(new_variable_name, dtype, dimension, fill_value=fill_value)
    dsout[new_variable_name][:] = values
    
    for k,v in attributes.items():
        if k == '_FillValue': continue
        new_variable.setncattr(k, v) # variable attr
    
    return dsout



def test_data_write(ds_new, title, comment, grid_size, 
                    core_variable_name, core_variable_standard_name, core_variable_unit, core_variable_dtype, core_variable_values,
                    lat_range=(0,None), lon_range=(0,None)):
    
    # set attribute
    now = dt.datetime.now()
    attr_dict = {'title' : title,
                 'grid' : f'{grid_size}',
                 'institution' : 'NIA',
                 'name_creator' : 'BNT',
                 'date_creation' : now.strftime('%Y-%m-%d %H:%M:%S'),
                 'comment' : comment}

    for k, v in attr_dict.items():
        ds_new.setncattr(k,v)

    lat_s, lat_e = lat_range
    lon_s, lon_e = lon_range
    
    lat_force_cut = None
    lon_force_cut = None
    
    if grid_size == 0.081 : 
        lat_force_cut = -1
        lon_force_cut = -1
    if grid_size == 0.054 :
        lat_force_cut = -1
        
        
    lat_grid = np.arange(-90 + (grid_size/2), 90 + (grid_size/2), grid_size)[:lat_force_cut][lat_s:lat_e]
    lon_grid = np.arange(0 + (grid_size/2), 360 + (grid_size/2), grid_size)[:lon_force_cut][lon_s:lon_e]
    
    # set dimension
    dim_dict = {'ntime' : 1,
                'nlat' : len(lat_grid),
                'nlon' : len(lon_grid)}

    for k, v in dim_dict.items():
        ds_new.createDimension(k,v)

    # set variables
    for variable_name in ['time', 'lat', 'lon', core_variable_name]:

        if variable_name == 'time' :
            variable_attribute = {'standard_name' : 'time',
                                  'format' : 'Mdd',
                                  'axis' : 'T'}
            dtype = np.int16
            dimensions = ('ntime',)
            variable_values = 101

        if variable_name == 'lat' :
            variable_attribute = {'standard_name' : 'latitude',
                                  'units' : 'degrees',
                                  'axis' : 'Y'}
            dtype = np.float32
            dimensions = ('nlat',)
            variable_values = lat_grid

        if variable_name == 'lon' :
            variable_attribute = {'standard_name' : 'longitude',
                                  'units' : 'degrees',
                                  'axis' : 'X'}
            dtype = np.float32
            dimensions = ('nlon',)
            variable_values = lon_grid
            
        if variable_name == core_variable_name :
            variable_attribute  = {'standard_name' : core_variable_standard_name,
                                   'units' : core_variable_unit}
            dtype = core_variable_dtype
            dimensions = ('ntime', 'nlat', 'nlon',)
            variable_values = core_variable_values[lat_s:lat_e, lon_s:lon_e]


        fill_value = -999

        ds_new = create_new_variable(ds_new,
                                     new_variable_name=variable_name,  
                                     dtype=dtype,
                                     dimension=dimensions,
                                     fill_value=fill_value,
                                     values=variable_values,
                                     attributes=variable_attribute)

    return ds_new