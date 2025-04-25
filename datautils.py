import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1','Solar','PEMS03','PEMS04','PEMS07','PEMS08', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange'
        ] # 原始作者預設支援多種 benchmark 資料集，包括交通、電力等知名公開資料集。 或許 可以用同一套主程式與模型結構，快速替換不同資料集。
# 1. ettm1, ettm2, etth1, etth2 ->	ETT 系列資料（電力需求、負載）
# 2. Solar, electricity ->	能源類資料
# 3. traffic, PEMS03~08 -> 交通路況資料
# 4. weather -> 天氣氣象資料
# 5. exchange -> 匯率資料（金融）
# 6. exchange -> 匯率資料（金融）

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = True

    if params.dset == 'ettm1': #  判斷目前指定的資料集是否為 'ettm1'
        root_path = 'datasets/ETT-small/' # 資料的資料夾路徑，表示原始的 ETTm1.csv 放在 datasets/ETT-small/ 裡。
        size = [params.context_points, 0, params.target_points] # size 定義輸入輸出長度
                                                                # context_points：輸入的歷史步數，例如過去 336 分鐘。
                                                                # 0：預留（目前沒使用，通常是預測前的空窗）預設不使用，即模型直接根據過去的資料預測未來資料。
                                                                # target_points：模型要預測未來幾點，例如未來 96 點。
                                                                #  xLSTM 這種結構可以直接輸入 context → 預測 target，因此中間 label 可省略。
        dls = DataLoaders( # 建立資料加載器
                datasetCls=Dataset_ETT_minute, # ETT 分鐘級資料專用
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features, # 'M' 表示多變量（multivariate）
                'scale': True, # 是否標準化
                'size': size, # 對應的輸入/輸出長度
                'use_time_features': params.use_time_features # 是否加上時間欄位（例如週期性特徵）
                },
                batch_size=params.batch_size, # 批次大小
                workers=params.num_workers, # 執行緒數
                ) # 給定 Dataset 所需的參數，包括資料檔名、標準化、是否加入時間特徵（如小時、週期）、資料切分長度（size），最後交由 DataLoaders() 包裝成 PyTorch 用的訓練與測試資料迭代器。



    elif params.dset == 'Solar':
        root_path = '/home/musleh/Downloads/iTransformer_datasets/Solar'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Solar,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'solar_AL.txt',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'PEMS04':
        root_path = '/home/musleh/Downloads/iTransformer_datasets/PEMS/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_PEMS, # PEMS 路況資料（.npz 格式）
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'PEMS04.npz',  #PEMS03.npz  data.npy
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'PEMS03':
        root_path = '/home/musleh/Downloads/iTransformer_datasets/PEMS/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_PEMS, # PEMS 路況資料（.npz 格式）
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'PEMS03.npz',  #PEMS03.npz  data.npy
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'PEMS07':
        root_path = '/home/musleh/Downloads/iTransformer_datasets/PEMS/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_PEMS, # PEMS 路況資料（.npz 格式）
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'PEMS07.npz',  #PEMS03.npz  data.npy
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'PEMS08':
        root_path = '/home/musleh/Downloads/iTransformer_datasets/PEMS/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_PEMS, # PEMS 路況資料（.npz 格式）
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'PEMS08.npz',  #PEMS03.npz  data.npy
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'ettm2':
        root_path = 'datasets/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        root_path = 'datasets/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        root_path = 'datasets/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    

    elif params.dset == 'electricity':
        root_path = 'datasets/electricity'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom, # 通用 CSV 格式
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        root_path = 'datasets/traffic'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom, # 通用 CSV 格式
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'weather':
        root_path = 'datasets/weather'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        root_path = 'datasets/illness'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom, # 通用 CSV 格式
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        root_path = 'datasets/exchange_rate'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom, # 通用 CSV 格式
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    #for i, batch in enumerate(dls.valid):
    #    print(i, len(batch), batch[0].shape, batch[1].shape)
    #breakpoint()
