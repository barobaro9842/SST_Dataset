from netCDF4 import Dataset
from netCDFfunc.preprocessing import *
from netCDFfunc.utility import *
from netCDFfunc.processing_tools import *

import argparse
import pandas as pd

def process_data(args):
    
    reference_data_path = args.reference_data_path
    raw_path = args.raw_path
    output_path = args.output_path
    
    k_day = args.k_day
    region = args.region
    dataset_names = args.dataset_names
    grid_set = args.grid_set
    
    if dataset_names == [] :
        dataset_names = ['AVHRR_OI_SST', 'CMC', 'DMI_SST', 'GAMSSA_GDS2', 'MUR_SST', 'MW_IR_SST', 'MW_OI_SST', 'NAVO_K10_SST_GDS2', 'OSPO_SST_Night', 'OSPO_SST', 'OSTIA_SST']
    if grid_set == []:
        grid_set = ['0.05', '0.054', '0.081', '0.08789', '0.1', '0.25'] #'0.01', 

    for ds_name in dataset_names :
        
        print(f'{ds_name} processing...')
        if ds_name == 'AVHRR_OI_SST':
            raw_ds_name = os.path.join(ds_name, 'v2.1')
        elif ds_name == 'CMC' :
            raw_ds_name = os.path.join(ds_name, 'deg0.1')
        else :
            raw_ds_name = ds_name

        # anomally and grade save location
        target_path_base = os.path.join(output_path, ds_name)
        
        if not os.path.exists(target_path_base) : 
            os.mkdir(target_path_base)
            
        # get recent 6 days' raw nc file date list

        # year folder problem ################################
        raw_data_path = os.path.join(raw_path, raw_ds_name, '2022')
        raw_data_name = os.listdir(raw_data_path)[-1][8:]
        
        raw_data_date_list = sorted(list(map(get_date, os.listdir(raw_data_path))))[-k_day:]
        
        # process data by date
        for date in raw_data_date_list:
            
            raw_data_file_name = date + raw_data_name
            
            # check path
            img_path = os.path.join(target_path_base, 'img')
            if not os.path.exists(img_path) : os.mkdir(img_path)
            
            nc_path = os.path.join(target_path_base, 'nc')
            if not os.path.exists(nc_path) : os.mkdir(nc_path)
            
            csv_path = os.path.join(target_path_base, 'csv')
            if not os.path.exists(csv_path) : os.mkdir(csv_path)
            
            anomaly_img_path, grade_img_path = get_path(img_path, date)
            anomaly_nc_path, grade_nc_path = get_path(nc_path, date)
            anomaly_csv_path, grade_csv_path = get_path(csv_path, date)
            
            raw_data_file_name_new = raw_data_file_name.replace('nc','')
            anomaly_img_file_path = os.path.join(anomaly_img_path, f'{raw_data_file_name_new}_{region}_anomaly.png')
            grade_img_file_path = os.path.join(grade_img_path, f'{raw_data_file_name_new}_{region}_grade.png')
            
            anomaly_nc_file_path = os.path.join(anomaly_nc_path, f'{raw_data_file_name_new}_{region}_anomaly.nc')
            grade_nc_file_path = os.path.join(grade_nc_path, f'{raw_data_file_name_new}_{region}_grade.nc')
            
            anomaly_csv_file_path = os.path.join(anomaly_csv_path, f'{raw_data_file_name_new}_{region}_anomaly.csv')
            grade_csv_file_path = os.path.join(grade_csv_path, f'{raw_data_file_name_new}_{region}_grade.csv')
            
            if os.path.exists(anomaly_img_file_path) and os.path.exists(grade_img_file_path) and os.path.exists(anomaly_nc_file_path) and os.path.exists(grade_nc_file_path):
                continue
            
            if int(date) <= 20201231 :
                period = 1
            else :
                period = 2
                
            ds_mean = dict()
            ds_ice = dict()
            ds_pctl = dict()
            
            # load base data
            for grid_size in grid_set :
                
                ds_mean[grid_size] = Dataset(f'{reference_data_path}/{period}/avg/{grid_size}/avg_{period}_{region}_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['avgsst'][:].data[0]
                ds_ice[grid_size] = Dataset(f'{reference_data_path}/{period}/ice/{grid_size}/ice_{period}_{region}_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['ice'][:].data[0]
                ds_pctl[grid_size] = Dataset(f'{reference_data_path}/{period}/pctl/{grid_size}/pctl_{period}_{region}_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['pctlsst'][:].data[0]
            
            # data processing
            meta_data_dic = get_meta_data(raw_data_path, date, raw_data_file_name)
            
            grid = meta_data_dic['grid']
            sst, ice = preprocessing_dataset(ds_name, meta_data_dic)
            
            sst = cropping(sst, region, grid)
            ice = cropping(ice, region, grid).astype(bool)
            
            if region == 'rok' and ds_name == 'MW_IR_SST':
                ice = ice[:-1,:]
                sst = sst[:-1,:]
            
            mean = ds_mean[str(grid)]
            pctl = ds_pctl[str(grid)]
            base_ice = ds_ice[str(grid)].astype(bool)
            
            ice = base_ice + ice

            anomaly = get_anomaly_grade(sst, ice, mean, pctl)
            grade = get_anomaly_grade(sst, ice, mean, pctl, is_grade=True)
            
            # img save
            to_img(anomaly, output_path=anomaly_img_file_path, is_anomaly=True, save_img=True)
            to_img(grade, output_path=grade_img_file_path, is_grade=True, save_img=True)
            
            # nc write
            to_nc(anomaly_nc_file_path, anomaly, period, region, grid, date)
            to_nc(grade_nc_file_path, grade, period, region, grid, date, is_grade=True)
            
            # nc to csv
            to_csv(anomaly_nc_file_path, anomaly_csv_file_path, include_null=False)
            to_csv(grade_nc_file_path, grade_csv_file_path, include_null=False)
    
if __name__ == '__main__' :
    
    volume_name = os.path.join('/Volumes', 'T7')
    #volume_name = 'D:'
    raw_path = os.path.join(volume_name, 'raw_data')
    output_path = os.path.join(volume_name, 'output')
    reference_data_path = os.path.join(volume_name, 'reference_data')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--reference_data_path', type=str, default=reference_data_path, help='path of 30 years reference files')
    parser.add_argument('--output_path', type=str, default=output_path, help='path of output files')
    parser.add_argument('--raw_path', type=str, default=raw_path, help='path of raw files')
    
    parser.add_argument('--k_day', type=int, default=0, help='check data from k day ago')
    parser.add_argument('--region', type=str, default='', help='one of [rok, nw, global]')
    parser.add_argument('--dataset_names', type=list, default=[], help='specify dataset by list')
    parser.add_argument('--grid_set', type=list, default=[], help='specify grid size by list')
    
    args = parser.parse_args()
    
    process_data(args)