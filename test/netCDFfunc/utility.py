from .preprocessing import get_data_A
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import cv2


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