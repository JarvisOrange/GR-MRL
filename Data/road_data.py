import os
import numpy as np
import pandas as pd
from numpy import copy
from torch.utils.data import Dataset, DataLoader
from utils import *


import warnings
warnings.filterwarnings('ignore')

dataset_name_dict = {
    "B": 'BAY',
    "L": 'LA',
    'C': 'CD',
    'S': 'SZ',
}

class RoadData():
    def __init__(self, cfg, data_name, stage='pretrain',  target_day=3, logger=None):
        super(RoadData, self).__init__()
        self.cfg = cfg
        self._logger = logger

        self.his_num = cfg['his_num']
        self.pred_num = cfg['pre_num']
        self.stage = stage

        self.target_day = target_day

        self.data_name = data_name
    
        self.load_data(stage, self.data_name)


    def  load_data(self, stage, dataset_name):
        if dataset_name == "CD" or "SZ":
            interpolate_flag = True
        else:
            interpolate_flag = False 

        self._logger.info('Loading dataset: {}'.format(dataset_name))
        
        # adjacency matrix
        self.adj = np.load('./raw_data/{}/matrix.npy'.format(dataset_name))

        # time series data 
        # L * N * D
        # D = 4 = speed, 1/288 * index, index, (weekday * 12 * 24) + index % 2016
        X = np.load()('./raw_data/{}/dataset_expand.npy'.format(dataset_name))


        if interpolate_flag:
            # 对于CD和SZ数据集，进行插值处理
            interp_X = torch.nn.functional.interpolate(X, size = 2 * X.shape[-1] - 1,mode='linear',align_corners=True)
            interp_X = torch.cat((interp_X[:,:,:1],interp_X),dim=-1)
            interp_X[:,1,0] = ((interp_X[:,1,1] - 1) + 2016 ) % 2016 # 7 * 24 * 12 = 2016 is the week slot
            X = interp_X


        if stage == 'pretrain':
            interval = 12 * 3 # 3 hours ??? to be confirmed
            x, y = self.generate_dataset(X, self.his_num, self.pred_num, interval)
            self._logger.info('{}_source_train : x shape : {}, y shape : {}'.format(dataset_name, x.shape, y.shape))
        
        if stage == 'time cluster':
            x = X
            self._logger.info('time clustering...')
            x, y = self.generate_dataset(X, self.his_num, self.pred_num, interval)
            self._logger.info('{}_time_cluster : x shape : {}, y shape : {}'.format(dataset_name, x.shape, y.shape)) 

        if stage == 'road cluster':
            x = X
            self._logger.info('Road clustering...')
            x, y = self.generate_dataset(X, self.his_num, self.pred_num, interval)
            self._logger.info('{}_road_cluster : x shape : {}, y shape : {}'.format(dataset_name, x.shape, y.shape)) 

        if stage == 'source_train':
            interval = 72 # a day to be confirmed
            x, y = self.generate_dataset(X, self.his_num, self.pred_num, interval)
            self._logger.info('{}_source_train : x shape : {}, y shape : {}'.format(dataset_name, x.shape, y.shape))

        if stage == 'target':
            interval = 72 # a day to be confirmed
            X = X[:, :, :288 * self.target_days] # 24 * 12 = 288
            x, y = self.generate_dataset(X, self.his_num, self.pred_num, interval)
            self._logger.info('{}_target : x shape : {}, y shape : {}'.format(dataset_name, x.shape, y.shape))    

        if stage == 'test':
            X = X[:, :, 288 * self.target_days:]
            interval = 72 # a day
            x, y = self.generate_dataset(X, self.his_num, self.pred_num, interval)
            self._logger.info('{}_test : x shape : {}, y shape : {}'.format(dataset_name, x.shape, y.shape))


    def generate_dataset(X, num_timesteps_input, num_timesteps_output, interval_step):
        # Generate the beginning index and the ending index of a sample, which
        # contains (num_points_for_training + num_points_for_predicting) points
        indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                in range(0, X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1, interval_step)]

        features, target = [], []
        for i, j in indices:
            features.append(
                # [N, 4, L]
                X[:, :, i: i + num_timesteps_input].transpose(
                    (0, 2, 1)))
            target.append(X[:, 0, i + num_timesteps_input: j])

        x = torch.from_numpy(np.array(features)).float()
        y = torch.from_numpy(np.array(target)).float()
        
        # x : [B, N, L, 4]
        # y : [B, N, L]
        return x,y
    
    def get_adj(self):
        """
        Get the adjacency matrix of the dataset
        """
        return self.adj