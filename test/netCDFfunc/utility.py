import re
from .preprocessing import get_data_A
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import numpy as np
from netCDF4 import Dataset

import datetime as dt
import os


days = [31,28,31,30,31,30,31,31,30,31,30,31]

def check_err_mask_A(variable_name):
    
    months = range(1,13)
    
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
                        


def to_img(arr, output_path, time, lon=None, lat=None, figsize=(), save_img=False, show_img=False, is_anomaly = False, is_grade=False):
    
    plt.style.use('default')
    
    if figsize == ():
        x, y = arr.shape
    else :
        x,y = figsize
        
    x = x/60
    y = y/60
    
    fig, ax = plt.subplots(figsize=(24,12))
    gca_ax = plt.gca()
    
    if arr.dtype == np.float32:
        np.place(arr, arr[:,:]==-999, np.nan)
        
    if is_anomaly == True :
        cmap = cm.jet.copy()
        vmax = 10.2
        vmin = 0
        
    elif is_grade == True :
        vmax = 5
        vmin = 0
        
        grade_0 = np.array([179/256, 241/256, 255/256, 1])
        grade_1 = np.array([255/256, 255/256, 128/256, 1])
        grade_2 = np.array([255/256, 179/256, 53/256, 1])
        grade_3 = np.array([255/256, 129/256, 0/256, 1])
        grade_4 = np.array([203/256, 76/256, 1/256, 1])
        grade_5 = np.array([153/256, 26/256, 0/256, 1])
        new_cmap = np.array([grade_0, grade_1, grade_2, grade_3, grade_4, grade_5])

        if 5 not in arr :
            new_cmap = new_cmap[:-1]
            if 4 not in arr :
                new_cmap = new_cmap[:-1]
                if 3 not in arr :
                    new_cmap = new_cmap[:-1]
                    if 2 not in arr :
                        new_cmap = new_cmap[:-1]
                        if 1 not in arr :
                            new_cmap = new_cmap[:-1]
            
        cmap = ListedColormap(new_cmap)
        
    cmap.set_bad(color='gray')
    cmap.set_under(color=np.array([250/256, 250/256, 250/256, 1]))
    
    if type(lat) != np.ndarray or type(lon) != np.ndarray :
        if is_anomaly == True : im = plt.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        elif is_grade == True : im = plt.imshow(arr, cmap=cmap, origin='lower', vmin=vmin)#, vmax=vmax)
    else :
        im = plt.imshow(arr, cmap=cmap, origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=vmin, vmax=vmax) 
    
        plt.xticks(range(0,361, 20))
        plt.yticks(range(-80,81,20))
        plt.grid(True, linestyle='--', color='black')
        
        x_labels = ['20°E','40°E','60°E','80°E','100°E','120°E','140°E','160°E','180°','160°W','140°W','120°W','100°W','80°W','60°W','40°W','20°W','0','20°E']
        y_labels = ['80°S','60°S','40°S','20°S','0°','20°N','40°N','60°N','80°N']
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
    
    divider = make_axes_locatable(gca_ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if is_anomaly == True : 
        plt.colorbar(im, cax=cax)
    
    plt.text(-30,0.9,f'{time}',{'fontsize':30}, transform=plt.gca().transAxes, va='top', ha='left')
    
    if save_img == True :
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show_img == True :
        plt.show()
    plt.close()


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

    result = []
    
    s_year, s_month, s_day = start_date
    e_year, e_month, e_day = end_date
        
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
        
    return np.array(result)
    
    
def get_data_by_date(base_dir, get_data_func, var_name ,start_date : tuple, end_date : tuple, specific_year=None, specific_date=(), is_mask=False) -> list:
    
    result_dic = dict()
    
    s_year, s_month, s_day = start_date
    e_year, e_month, e_day = end_date
    
    if specific_date != (): specific_month, specific_day = specific_date
    else : specific_month, specific_day = None, None
        
    for year in tqdm(range(s_year, e_year+1)) :
        
        if specific_year != None and year != specific_year : continue
        
        if year == s_year : months = range(s_month, 13)
        elif year == e_year : months = range(1, e_month+1)
        else : months = range(1,13)
        
        for month, day_len in zip(months, days[months[0]-1:]):
            
            if specific_month != None and month != specific_month : continue
            
            if year == s_year and month == s_month : day_range = range(s_day, day_len+1)
            elif year == e_year and month == e_month : day_range = range(1, e_day+1)
            else : day_range = range(1, day_len+1)
            
            for day in day_range :
                
                if specific_day != None and day != specific_day : continue
                
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
                arr = np.array(input[(month, day)])
                pctl_arr = np.percentile(arr, 90 ,axis=0)
                result_dic[(month, day)] = pctl_arr.tolist()
                
    
    return result_dic


def copy_existing_variable(ds_in, ds_out, default_fill_value):
    
    for attr in ds_in.ncattrs() :
        
        if attr == 'anom' : continue
        elif attr == 'err' : new_attr = 'analysis_error'
        elif attr == 'ice' : new_attr = 'sea_ice_fraction'
        elif attr == 'sst' : new_attr = 'analysed_sst'
        else : new_attr = attr
        
        ds_out.setncattr(new_attr, ds_in.getncattr(attr))

    for k, v in ds_in.dimensions.items():
        ds_out.createDimension(k, v.size)
    
    for name, variable in ds_in.variables.items():
    
        existing_variable = ds_out.createVariable(name, variable.datatype, variable.dimensions, fill_value=default_fill_value) # name, datatype, dimension
        ds_out[name][:] = ds_in[name][:] # values

        for attr in variable.ncattrs():
            if attr == '_FillValue': continue
            existing_variable.setncattr(attr, variable.getncattr(attr)) # variable attr
                
    return ds_out


def create_new_variable(ds_out, new_variable_name, dtype, dimension, fill_value, values, attributes):
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
            
    new_variable = ds_out.createVariable(new_variable_name, dtype, dimension, fill_value=fill_value)
    ds_out[new_variable_name][:] = values
    
    for k,v in attributes.items():
        if k == '_FillValue': continue
        new_variable.setncattr(k, v) # variable attr
    
    return ds_out



def nc_write(ds_new, title, comment, grid_size, 
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

def masking(input, mask, fill_value):
    result = []
    if len(input.shape) == 3 and len(input.shape) == 3 :
        for arr, m in zip(input, mask):
            arr = np.ma.array(arr, mask = m)
            arr = arr.filled(fill_value = fill_value)
            result.append(arr)
        return np.array(result)
    
    elif len(input.shape) == 3 and len(input.shape) == 2 :
        for arr in input :
            arr = np.ma.array(arr, mask = mask)
            arr = arr.filled(fill_value = fill_value)
            result.append(arr)
        return np.array(result)
    
    elif len(input.shape) == 2 :
        arr = input
        arr = np.ma.array(arr, mask = mask)
        arr = arr.filled(fill_value = fill_value)
        
        return np.array(arr)
        
    
    
def cropping(arr, region):
    if region == 'rok':
        return arr[440:572, 440:600]
    if region == 'nw':
        return arr[280:624, 392:1136]
    if region == 'global':
        return arr
    
    
def get_anomaly_heatlevel(ds_in, ds_sst, ds_ice, is_heat_level=False) :
    '''
    generator
    is_heat_level = False : return anomaly
    is_heat_level = True : return heat_level
    ds from get_data_by_date (dict{(month,day) : data} type data)
    '''

    for month, day_len in tqdm(zip(range(1,13), days)):
        for day in range(1,day_len+1):
            
            date = dt.date(1000, month,day).strftime('%m%d')

            pctl = np.percentile(ds_in[(month, day)], 90, axis=0)
            pctl = cropping(pctl, 'rok')
            np.place(pctl, pctl[:,:]==-999, np.nan)
            
            mean = np.mean(ds_in[(month, day)], axis=0)
            mean = cropping(mean, 'rok')
            np.place(mean, mean[:,:]==-999, np.nan)
            
            sst = ds_sst[(month, day)][0]
            sst = cropping(sst, 'rok')
            
            ice_accum = np.sum(ds_ice[(month, day)], axis=0)
            np.place(ice_accum, ice_accum[:,:] <= 30 * 0.15, False)
            np.place(ice_accum, ice_accum[:,:] > 30 * 0.15, True)
            ice_accum = ice_accum.astype(bool)
            ice_accum = cropping(ice_accum, 'rok')
            
            anomaly = sst - mean
            np.place(anomaly, anomaly[:,:]<0, 0)
            anomaly = masking(anomaly, np.invert(ice_accum), fill_value=-1)
            
            if is_heat_level == False:
                yield (month,day), anomaly

            
            elif is_heat_level == True :
                diff = pctl - mean
                
                heat_level = np.ceil(anomaly / diff)
                heat_level = masking(heat_level, np.invert(ice_accum), fill_value=-1)
                np.place(heat_level, heat_level[:,:]>5, 5)
            
                yield (month,day), heat_level
            
            # save_img(anomaly, f'/Volumes/T7/intermediate_output/rok_anomaly_heatlevel_criteria_change_2/anomaly_{date}', date)
            # save_img(heat_level, f'/Volumes/T7/intermediate_output/rok_anomaly_heatlevel_criteria_change_2/heat_level_{date}', date, is_grade=True)