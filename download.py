import numpy as np

import os

import shutil
import requests
import datetime

import argparse

import signal
import time


def download_data(args):
    ''' 
    start_date and end_date = Tuple (Year, Month, Day)
    
     1) start_date and end_date == None : most_recent
     2) start_date != None and end_date == None : start_date ~ most_recent
     3) start_date != None and end_date != None : start_date ~ end_date

    dataset_names = list of (AVHRR, CMC, DMI, GAMSSA, MUR25, MUR0.01, MWIR, MW, NAVO, OSPON, OSPO, OSTIA)
    '''
    
    common_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/ghrsst/data'
    
    download_path = args.download_path
    k_day = args.k_day
    dataset_names = args.dataset_names

    if dataset_names == [] :
        dataset_names = ['AVHRR', 'CMC', 'DMI', 'GAMSSA', 'MUR25', 'MUR', 'MWIR', 'MW', 'NAVO', 'OSPON', 'OSPO', 'OSTIA']

    # if start_date == None and end_date == None : 
    
    date_range = range(k_day,0,-1)
    end_date = datetime.datetime.now()
    
    # else: 
    #     start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    #     end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    #     date_range = range((end_date - start_date).days, -1, -1)
        
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
                min_file_size  = 1000 * 1024
            
            elif dataset_name == 'CMC' :
                file_name = f'{date}120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/CMC/CMC0.1deg/v3/{year}/{j_day}/{file_name}'
                min_file_size = 6000 * 1024
                
            elif dataset_name == 'DMI' :
                file_name = f'{date}000000-DMI-L4_GHRSST-SSTfnd-DMI_OI-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/DMI/DMI_OI/v1/{year}/{j_day}/{file_name}'
                min_file_size = 150279 * 1024
                
            elif dataset_name == 'GAMSSA' :
                file_name = f'{date}120000-ABOM-L4_GHRSST-SSTfnd-GAMSSA_28km-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/ABOM/GAMSSA/v1.0/{year}/{j_day}/{file_name}'
                min_file_size = 1154 * 1024
                
            elif dataset_name == 'MUR25' :
                file_name = f'{date}090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc'
                url = f'{common_url}/GDS2/L4/GLOB/JPL/MUR25/v4.2/{year}/{j_day}/{file_name}'
                min_file_size = 1908 * 1024
                
            elif dataset_name == 'MUR' :
                file_name = f'{date}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
                url = f'{common_url}/GDS2/L4/GLOB/JPL/MUR/v4.1/{year}/{j_day}/{file_name}'
                min_file_size = 700 * 1024 * 1024
                
            elif dataset_name == 'MWIR' :
                file_name = f'{date}120000-REMSS-L4_GHRSST-SSTfnd-MW_IR_OI-GLOB-v02.0-fv05.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/REMSS/mw_ir_OI/v5.0/{year}/{j_day}/{file_name}'
                min_file_size = 5800 * 1024
                
            elif dataset_name == 'MW' :
                file_name = f'{date}120000-REMSS-L4_GHRSST-SSTfnd-MW_OI-GLOB-v02.0-fv05.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/REMSS/mw_OI/v5.0/{year}/{j_day}/{file_name}'
                min_file_size = 594 * 1024
                
            elif dataset_name == 'NAVO' :
                file_name = f'{date}000000-NAVO-L4_GHRSST-SST1m-K10_SST-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/NAVO/K10_SST/v1/{year}/{j_day}/{file_name}'
                min_file_size = 38018 * 1024
                
            elif dataset_name == 'OSPON' :
                file_name = f'{date}000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended_Night-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/OSPO/Geo_Polar_Blended_Night/v1/{year}/{j_day}/{file_name}'
                min_file_size = 19000 * 1024
                
            elif dataset_name == 'OSPO' :
                file_name = f'{date}000000-OSPO-L4_GHRSST-SSTfnd-Geo_Polar_Blended-GLOB-v02.0-fv01.0.nc'
                url = f'{common_url}/GDS2/L4/GLOB/OSPO/Geo_Polar_Blended/v1/{year}/{j_day}/{file_name}'
                min_file_size = 19000 * 1024
                
            elif dataset_name == 'OSTIA' :
                file_name = f'{date}-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIA.nc.bz2'
                url = f'{common_url}/L4/GLOB/UKMO/OSTIA/{year}/{j_day}/{file_name}'
                min_file_size = 151921 * 1024

            if file_name.endswith('.bz2') : file_name = file_name.replace('.bz2', '')
            target_file = os.path.join(download_path, dataset_name, year, file_name)
            
            
            if os.path.exists(target_file) and os.path.getsize(target_file) > min_file_size * 0.95:
                print(f'[{date}|{dataset_name}] is already downloaded!')
                continue
            
            response = requests.get(url, stream=True)
            
            if response.status_code == 404:
                print(f'[{date}|{dataset_name}] doesn\'t exist!')
                continue
            
            d_path = os.path.join(download_path, dataset_name)
            if not os.path.exists(d_path) : os.mkdir(d_path)
                
            y_path = os.path.join(d_path, year)
            if not os.path.exists(y_path) : os.mkdir(y_path)
            
            
            with open(target_file, 'wb') as out_file:
                print(f'[{date}|{dataset_name}] downloading...')
                shutil.copyfileobj(response.raw, out_file)
                
            del response    

class TimeOutException(Exception) :
    pass

def alarm_handler(signum, frame):
    print("timeout")
    raise TimeOutException()
            
if __name__ == '__main__' :
    
    download_path = '/Volumes/T7/download_data'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--download_path', type=str, default=download_path, help='download directory')
    parser.add_argument('--k_day', type=int, default=0, help='check data from k day ago')
    parser.add_argument('--dataset_names', type=list, default=[], help='specify dataset by list')

    args = parser.parse_args()
    
    time_out_limit = args.k_day * 20 * 60 # 20minutes for a day
    
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(time_out_limit)
    
    try :
        download_data(args)
    except TimeOutException as e:
        print(e)