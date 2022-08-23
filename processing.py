from netCDF4 import Dataset
from netCDFfunc.preprocessing import *
from netCDFfunc.utility import *
from netCDFfunc.processing_tools import *
import argparse

# def to_nc(args, data, grid, date):
    
#     output_dir = args.output_path
#     variable_name = 'anomalysst'
#     variable_standard_name = 'SST Anomaly'
#     variable_unit = 'degree C'
#     variable_dtype = np.float32

#     file_name = f'test' 
    
#     nc_path = os.path.join(output_dir,file_name+'.nc')
#     ds_new = Dataset(nc_path, 'w', format='NETCDF4')

#     title = f'ROK 30 years(1991~2020) base SST anomaly data ({date})' 
#     comment = f'Average SST calculated 1991/1/1~2020/12/31'
    
#     variable_values = data
#     lat_range = (440, 572)
#     lon_range = (440, 600)

#     ds_new = nc_write(ds_new, title, comment, grid, 
#                         variable_name, variable_standard_name, variable_unit, variable_dtype, variable_values, 
#                         lat_range, lon_range)
#     ds_new.close()

def data_processing(args):
    
    base_data_path = args.base_data_path
    raw_path = args.raw_path
    output_path = args.output_path
    
    k_day = args.k_day
    region = args.region
    dataset_names = args.dataset_names
    grid_set = args.grid_set
    
    if dataset_names == [] :
        dataset_names = ['AVHRR', 'CMC', 'DMI', 'GAMSSA', 'MUR25', 'MUR', 'MWIR', 'MW', 'NAVO', 'OSPON', 'OSPO', 'OSTIA']
    if grid_set == []:
        grid_set = ['0.01', '0.05', '0.054', '0.081', '0.08789', '0.1', '0.25']

    for ds_name in dataset_names :
        
        print(f'{ds_name} processing...')
            
        # anomally and grade save location
        target_path_base = os.path.join(output_path, ds_name)
        
        if not os.path.exists(target_path_base) : 
            os.mkdir(target_path_base)
            
        # get recent 6 days' raw nc file date list
        raw_data_path = os.path.join(raw_path, ds_name)
        raw_data_name = os.listdir(raw_data_path)[-1][8:]
        
        raw_data_date_list = sorted(list(map(get_date, os.listdir(raw_data_path))))[-k_day:]
        
        # process data by date
        for date in raw_data_date_list:
            
            raw_data_file_name = date + raw_data_name
            
            # check path
            img_path = os.path.join(target_path_base, 'img')
            if not os.path.exists(img_path) : os.mkdir(img_path)
            
            anomaly_img_path, grade_img_path = get_path(img_path, date)
            
            anomaly_img_file_path = os.path.join(anomaly_img_path, f'{raw_data_file_name}_{region}_anomaly.png')
            if os.path.exists(anomaly_img_file_path):
                continue
            
            grade_img_file_path = os.path.join(grade_img_path, f'{raw_data_file_name}_{region}_grade.png')
            if os.path.exists(grade_img_file_path):
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
                
                ds_mean[grid_size] = Dataset(f'{base_data_path}/{period}/avg/{grid_size}/avg_{period}_{region}_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['avgsst'][:].data[0]
                ds_ice[grid_size] = Dataset(f'{base_data_path}/{period}/ice/{grid_size}/ice_{period}_{region}_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['ice'][:].data[0]
                ds_pctl[grid_size] = Dataset(f'{base_data_path}/{period}/pctl/{grid_size}/pctl_{period}_{region}_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['pctlsst'][:].data[0]
            
            # check ouput data status
            # output_date_list = sorted(list(map(get_date, os.listdir(target_path_base))))[-k_day:]
            # target_path = os.path.join(target_path_base, date)
            # if date not in output_date_list and not os.path.exists(target_path):
            #     os.mkdir(target_path)
                
            # overlap check
            # if len(os.listdir(target_path)) == 4 : continue
            
            # data processing
            meta_data_dic = get_meta_data(raw_data_path, date, raw_data_file_name)
            
            grid = meta_data_dic['grid']
            sst, ice = preprocessing_dataset(ds_name, meta_data_dic)
            
            sst = cropping(sst, region, grid)
            ice = cropping(ice, region, grid).astype(bool)
            
            if region == 'rok' and ds_name == 'MWIR':
                ice = ice[:-1,:]
                sst = sst[:-1,:]
            
            mean = ds_mean[str(grid)]
            pctl = ds_pctl[str(grid)]
            base_ice = ds_ice[str(grid)].astype(bool)
            
            ice = base_ice + ice

            # img save
            anomaly = get_anomaly_grade(sst, ice, mean, pctl)
            np.place(anomaly, anomaly[:,:] > 10000, 32767)
            
            to_img(anomaly, output_path=anomaly_img_file_path, is_anomaly=True, save_img=True)
            np.place(anomaly, anomaly[:,:] == 32767, np.nan)
            
            grade = get_anomaly_grade(sst, ice, mean, pctl, is_grade=True)
            to_img(grade, output_path=grade_img_file_path, is_grade=True, save_img=True)
            np.place(grade, grade[:,:] == 10, np.nan)
            
            
            # nc write
            # to_nc(args, anomaly, grid, date)
            
            
if __name__ == '__main__' :
    
    raw_path = '/Volumes/T7/download_data'
    output_path = '/Volumes/T7/output_data'
    base_data_path = '/Volumes/T7/base_data'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_data_path', type=str, default=base_data_path, help='path of 30 years base files')
    parser.add_argument('--output_path', type=str, default=output_path, help='path of output files')
    parser.add_argument('--raw_path', type=str, default=raw_path, help='path of raw files')
    
    parser.add_argument('--k_day', type=int, default=0, help='check data from k day ago')
    parser.add_argument('--region', type=str, default='', help='one of [rok, nw, global]')
    parser.add_argument('--dataset_names', type=list, default=[], help='specify dataset by list')
    parser.add_argument('--grid_set', type=list, default=[], help='specify grid size by list')
    
    args = parser.parse_args()
    
    data_processing(args)