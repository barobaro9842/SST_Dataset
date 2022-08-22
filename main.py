from netCDF4 import Dataset
from netCDFfunc.preprocessing import *
from netCDFfunc.utility import *
from netCDFfunc.processing import *



def data_processing(base_path, download_path, output_path, dataset_names=[], grid_set=[]):
    
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
        raw_data_path = os.path.join(download_path, ds_name)
        raw_data_file_name = os.listdir(raw_data_path)[-1][8:]
        
        raw_data_date_list = sorted(list(map(get_date, os.listdir(raw_data_path))))[-10:]
        
        # process data by date
        for date in raw_data_date_list:
            
            if int(date) <= 20201231 :
                period = 1
            else :
                period = 2
                
            ds_mean = dict()
            ds_ice = dict()
            ds_pctl = dict()
            
            # load base data
            for grid_size in grid_set :
                ds_mean[grid_size] = Dataset(f'{base_path}/{period}/avg/{grid_size}/avg_{period}_rok_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['avgsst'][:].data[0]
                ds_ice[grid_size] = Dataset(f'{base_path}/{period}/ice/{grid_size}/ice_{period}_rok_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['ice'][:].data[0]
                ds_pctl[grid_size] = Dataset(f'{base_path}/{period}/pctl/{grid_size}/pctl_{period}_rok_{date[4:]}_{grid_size}.nc', 'r', format='NETCDF4').variables['pctlsst'][:].data[0]
            
            # check ouput data status
            output_date_list = sorted(list(map(get_date, os.listdir(target_path_base))))[-6:]
            target_path = os.path.join(target_path_base, date)
            if date not in output_date_list and not os.path.exists(target_path):
                os.mkdir(target_path)
                
            # overlap check
            if len(os.listdir(target_path)) == 4 : continue
            
            # data processing
            meta_data_dic = get_meta_data(raw_data_path, date, raw_data_file_name)
            
            grid = meta_data_dic['grid']
            sst, ice = preprocessing_dataset(ds_name, meta_data_dic)
            
            sst = cropping(sst, 'rok', grid)
            ice = cropping(ice, 'rok', grid).astype(bool)
            
            if ds_name == 'MWIR':
                ice = ice[:-1,:]
                sst = sst[:-1,:]
            
            mean = ds_mean[str(grid)]
            pctl = ds_pctl[str(grid)]
            base_ice = ds_ice[str(grid)].astype(bool)
            
            ice = base_ice + ice
        
            anomaly = get_anomaly_grade(sst, ice, mean, pctl)
            grade = get_anomaly_grade(sst, ice, mean, pctl, is_grade=True)
            
            to_img(anomaly, output_path=target_path, is_anomaly=True, save_img=True)
            
            
if __name__ == '__main__' :
    
    download_path = '/Volumes/T7/download_data'
    output_path = '/Volumes/T7/output_data'
    base_path = '/Volumes/T7/base_data'
    
    data_processing(base_path, download_path, output_path)