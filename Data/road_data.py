import os
import numpy as np
import pandas as pd
from numpy import copy
from torch.utils.data import Dataset, DataLoader

from utils import *
from Node_Embed_Model.Laplacian import *
from Node_Embed_Model.SpaceSyntax import *

import warnings
warnings.filterwarnings('ignore')


class RoadData():
    def __init__(self, cfg, name, flag='pretrain',  target_day=3, logger=None):
        super(RoadData, self).__init__()
        self.cfg = cfg
        self._logger = logger

        self.flag = flag
        self.target_day = target_day
        self.name = name
        
        self.load_data(flag, self.name)

    def load_data(self, flag, dataset_name):

        self._logger.info('Loading dataset: {}'.format(dataset_name))
        
        # adjacency matrix
        self.adj = np.load('./Raw_Data/{}/matrix.npy'.format(dataset_name))
        self.road_num = self.adj.shape[0]

        # time series data 
        # L * N * D
        # D = 7 = [speed, index_time_step, week_time, node, city, means, std]
        X = np.load('./Raw_Data/{}/dataset_new.npy'.format(dataset_name))


        # [N, 7, L]
        X = np.transpose(X, (1, 2, 0))

        self.mean, self.std = self.cfg['dataset_info'][self.name]['mean'], self.cfg['dataset_info'][self.name]['std']

        # norm
        X[:,0,:] = (X[:,0,:] - self.mean ) / self.std

        # turn time index(0, 1, 2,... ) to (0, 1/288, 2/288,...)
        X[:,1,:] = X[:,1,:] /  (12 * 24)
            
        
        if flag == 'pretrain': # use three source datasets only
            # [N, 7, L]
            self.his_num = 288
            self.pre_num = 0

            self.interval = 12 * 24 # all his, not predict 

            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)
        
        if flag == 'time_cluster':
            X = X
            self.his_num = 12
            self.pre_num = 0

            self.interval = 12

            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)

        if flag == 'road_cluster':
            X = X
            self.his_num = 12
            self.pre_num = 0

            self.interval = 12

            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)

        if flag == 'rag':
            X = X
            self.his_num = 12
            self.pre_num = 0

            self.interval = 12

            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)
            
            
        if flag == 'source_train':
            self.his_num = self.cfg['his_num']
            self.pre_num = self.cfg['pre_num']

            self.interval = self.cfg['his_num']

            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)

        if flag == 'target_train':
            X = X[:, :, :288 * self.target_day] # 24 * 12 = 288

            self.his_num = self.cfg['his_num']
            self.pre_num = self.cfg['pre_num']

            self.interval = self.cfg['his_num']
            
            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)

        if flag == 'test':
            X = X[:, :, 288 * self.target_day:]

            self.his_num = self.cfg['his_num']
            self.pre_num = self.cfg['pre_num']

            self.interval = self.cfg['his_num']

            self.x, self.y = self.generate_data(X, self.his_num, self.pre_num, self.interval)


        assert self.x is not None, "x is None"
        if self.y is None:
            self._logger.warning("y is None, return x only")
            
        ################################################################################
        # origin self.x: [S, N, 7, l_his]
        # if self.flag == 'pretrain':
        #     # x : [S, N, 7, l_his]
        #     self._logger.info('{}_pretrain : x shape : {}, y shape : {}'.format(dataset_name, self.x.shape, self.y.shape))

        if self.flag == 'time_cluster':
            # x : [S,  N, 2, l_his]
            self.x = self.x[:, :, :2, :]
            self._logger.info('{}_time_cluster : x shape : {}, y shape : {}'.format(dataset_name, self.x.shape, self.y.shape)) 
            
        if self.flag == 'road_cluster':
            # x : [S, N, 2, l_his]
            self.x = self.x[:, :, :2, :]
            self._logger.info('{}_road_cluster : x shape : {}, y shape : {}'.format(dataset_name, self.x.shape, self.y.shape)) 

        if self.flag == 'rag':
            # x : [S, N, 2, l_his]
            self.x = self.x[:, :, :2, :]
            # x : [S* N,1, 2, l_his]
            self.x = self.x.reshape(-1, 1, 2, self.his_num)
            self._logger.info('{}_rag : x shape : {}, y shape : {}'.format(dataset_name, self.x.shape, self.y.shape))
            
        if self.flag == 'source_train' or self.flag == 'target_train' or self.flag == 'test':
            # x : [S * N, 1, 7, l_his]
            # y : [S * N, l_pre] 

            self.x = self.x.reshape(-1, 1, 7, self.his_num)
            self.y = self.y.reshape(-1, self.pre_num)

            self._logger.info('{}_{} : x shape : {}, y shape : {}'.format(dataset_name, self.flag, self.x.shape, self.y.shape))

    def get_data(self):
        """
        Get the data of the dataset
        """
        assert self.x is not None, "x is None"

        if self.y is None:
            self._logger.warning("y is None, return x only")
            

        # if self.flag == 'pretrain':
        #     # x : [S * N, l_his， 7]

        #     return self.x
        
        if self.flag == 'time_cluster':
            # x : [S, N, 2, l_his]
            return self.x

        if self.flag == 'road_cluster':
            # x : [S, N, 2, l_his]
            return self.x
        
        if self.flag == 'rag':
            # x : [S* N,1, 2, l_his]
            return self.x
        
        if self.flag == 'source_train' or self.flag == 'target_train' or self.flag == 'test':
            # x : [S * N, 1, 7, l_his]
            # y : [S * N, l_pre] 
            return self.x, self.y

    def generate_data(self, X, num_timesteps_input, num_timesteps_output, interval_step):
        # Generate the beginning index and the ending index of a sample, which
        # contains (num_points_for_training + num_points_for_predicting) points
        indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                in range(0, X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1, interval_step)]

        features, target = [], []
        for i, j in indices:
            temp = X[:, :, i: i + num_timesteps_input]
            # [N, 7, L]
            features.append(temp)

            #target unnorm
            target.append(X[:, 0, i + num_timesteps_input: j] * self.std + self.mean)
        
        x = np.array(features)
        y = np.array(target)

        # x : [S, N, l_his， 7]
        # y : [S, N, l_pre]
        return x, y
    
    def get_x_num(self):
        return self.x.shape[0]

    def get_road_num(self):
        return self.road_num
    
    def get_data_info(self):
        return self.his_num, self.pre_num, self.interval

    def get_adj(self):
        """
        Get the adjacency matrix of the dataset
        """
        return self.adj

    
    def get_node_embed(self):
        temp1 =  self.get_laplace_matrix()
        temp2 = self.get_space_syntax_matrix()
        return np.hstack((temp1,temp2))


    def get_laplace_matrix(self):
        return get_laplac_embed(self.adj, dim=self.cfg['laplacian_dim'])


    def get_space_syntax_matrix(self):
        return get_space_syntax_embed(self.adj)

    
    