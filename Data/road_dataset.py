from torch.utils.data import Data, Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from utils import *

from road_data import RoadData


# pretrain dataset(mask) flag pretrain
# cluster dataset(all) flag cluster
# cluster dataset(traffic) flag cluster
# rag_dataset flag rag
# finetune dataset flag finetune
# finetune dataset(test)
# try flag try

dataset_name_dict = {
    "B": 'BAY',
    "L": 'LA',
    'C': 'CD',
    'S': 'SZ',
}
class RoadDataset(Dataset):
    def __init__(self, flag, X, Y=None):
        super(RoadDataset,self).__init__()
        self.X = X
        self.Y = Y
        self.flag = flag


    def __len__(self):
        return self.X.shape

    def __getitem__(self, index):
        if self.flag == 'finetune':
            pass

        if self.flag == 'time_cluster':
            x= self.X[index]
            x = torch.from_numpy(x).float()
            return x


        if self.flag == 'road_cluster':
            x= self.X[index]
            x = torch.from_numpy(x).float()
            return x
            

        if self.flag == 'pretrain':
            x= self.X[index]
            y =self.Y[index]
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            return x, y
            
        if self.flag == 'rag':
            pass

        if self.flag =='try':
            pass

        x_data = x_data.float()
        y_data = y_data.float()
        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data,means=self.means_list[select_dataset],stds = self.stds_list[select_dataset])
        data_i.data_name = select_dataset
        
        return data_i


class RoadDataProvider():
    def __init__(self, cfg, flag, logger=None):
        super(RoadDataProvider, self).__init__()
        
        self.flag = flag
        self,cfg = cfg
        

        self.source_data, self.target_data = cfg['target_data'].split('_')
        self.source_data = self.source_data.split('-')

        self.target_data = dataset_name_dict[self.target_data]
        self.source_data = [dataset_name_dict[n] for n in self.source_data]

        if flag == 'source_train' or flag == 'target_train' or flag == 'test':
            self.data_list = [RoadData(cfg, self.source_data[i], flag=self.flag, logger=logger) for i in range(3)]

            X_num = 0, X_num_dict = {}
        
            his_num, pre_num, interval = self.data_list[0].get_data_info()
            
            for dataset in self.data_src_list:
                temp = dataset.get_data_num()
                X_num += temp
                X_num_dict[dataset.name] = temp
                
            self.X = np.zeros([X_num, his_num], dtype=float)
            self.Y = np.zeros([X_num, pre_num], dtype=float)

            cur = 0
            for dataset in self.data_list:
                x, y = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :] = x
                self.Y[cur:cur + X_num_dict[dataset.name], :] = y

                cur += dataset.get_data_num()

            #target test apply different epoch

        if flag == 'pretrain':
            self.data_src_list = [RoadData(cfg, self.source_data[i], flag='pretrain', logger=logger) for i in range(3)]

            X_num = 0, X_num_dict = {}
        
            his_num, pre_num, interval = self.data_src_list[0].get_data_info()
            
            for dataset in self.data_src_list:
                temp = dataset.get_data_num()
                X_num += temp
                X_num_dict[dataset.name] = temp
                
            self.X = np.zeros([X_num, his_num], dtype=float)
            self.Y = np.zeros([X_num, pre_num], dtype=float)

            cur = 0
            for dataset in self.data_src_list:
                x, y = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :] = x
                self.Y[cur:cur + X_num_dict[dataset.name], :] = y

                cur += dataset.get_data_num()

        if flag == 'time_cluster':
            ### time patch
            self.data_time_cluster_list = [RoadData(cfg, self.source_data[i], flag='time_cluster', logger=logger) for i in range(3)]

            X_num = 0, X_num_dict = {}
        
            his_num, pre_num, interval = self.data_src_list[0].get_data_info()
            
            for dataset in self.data_src_list:
                temp = dataset.get_data_num()
                X_num += temp
                X_num_dict[dataset.name] = temp
                
            self.X = np.zeros([X_num, his_num, 5], dtype=float)

            cur = 0
            for dataset in self.data_src_list:
                x = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :] = x

                cur += X_num_dict[dataset.name]
        
        if flag == 'road_cluster':
            ### road
            self.data_road_cluster_0 = RoadData(cfg, self.source_data[0], flag='road_cluster', logger=logger)
            self.data_road_cluster_1 = RoadData(cfg, self.source_data[1], flag='road_cluster', logger=logger)
            self.data_road_cluster_2 = RoadData(cfg, self.source_data[2], flag='road_cluster', logger=logger)
            self.data_road_cluster_list = [self.data_road_cluster_0, self.data_road_cluster_1, self.data_road_cluster_2]
            
            R_num = 0, R_num_dict = {}

            
            for dataset in self.data_road_cluster_list:
                temp = dataset.get_road_num()
                R_num += temp
                R_num_dict[dataset.name] = temp
            
            self.Road_E = np.zeros([R_num, self.cfg['laplacian_dim']+3], dtype=float)

            cur = 0
            for dataset in self.data_road_cluster_list:
                node_e = dataset.get_node_embed()
                
                self.Road_E[cur:cur + R_num_dict[dataset.name], :] = node_e
    
                cur += R_num_dict[dataset.name]

            cluster_label = kmeans(self.Road_E, self.cfg['road_k']) #[label_2, label_10..., label_1]

            cluster_label_node_dict = {}
            self.X_road_cluster_dict = {}

            self.X_road_cluster_0 = self.data_road_cluster_0.get_data()
            self.X_road_cluster_1 = self.data_road_cluster_1.get_data()
            self.X_road_cluster_2 = self.data_road_cluster_2.get_data()
            
            for k in range(self.cfg['road_k']):
                cluster_label_node_dict[k] = np.where(cluster_label == k)

            dataset_name_dict = R_num_dict.keys()
            pre, _, _= self.data_road_cluster_0.get_data_info()
            for k in cluster_label_node_dict.keys():
                temp_list = list(cluster_label_node_dict[k])
                temp_x = []
                for i in temp_list:
                    if i < R_num_dict[dataset_name_dict[0]]:
                        temp_x += list(np.squeeze(self.X_road_cluster_0[i], dim=0))
                    elif i < R_num_dict[dataset_name_dict[1]]:
                        temp_x += list(np.squeeze(self.X_road_cluster_1[i], dim=0))
                    elif i < R_num_dict[dataset_name_dict[2]]:
                        temp_x += list(np.squeeze(self.X_road_cluster_2[i], dim=0))
                temp_x = np.array(temp_x)
                self.X_road_cluster_dict[k] = temp_x #[, B*l, 5]
            

        if flag == 'rag':
            self.data_src_list = [RoadData(cfg, self.source_data[i], flag='pretrain', logger=logger) for i in range(3)]

            X_num = 0, X_num_dict = {}
        
            his_num, pre_num, interval = self.data_src_list[0].get_data_info()
            
            for dataset in self.data_src_list:
                temp = dataset.get_data_num()
                X_num += temp
                X_num_dict[dataset.name] = temp
                
            self.X = np.zeros([X_num, his_num], dtype=float)


            cur = 0
            for dataset in self.data_src_list:
                x = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :] = x

                cur += dataset.get_data_num()

        if flag =='try':
            pass
    

    def generate_dataloader(self):
        bs = self.cfg['stage']['self.flag']['batch_size']
        drop_last = self.cfg['drop_last']
        R_dataset = RoadDataset(self.flag, self.X, self.Y)

        dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)

        return dataloader

    def generate_road_cluster_dataloader(self):
        
        assert self.flag == 'road_cluster', 'this provider is not for road clustering'
        bs = self.cfg['pretrain']['batch_size']
        drop_last = self.cfg['drop_last']

        dataloader_list = []

        for k in self.X_road_cluster_dict.keys():
            R_dataset = RoadDataset(self.flag, self.X_road_cluster_dict[k])
            dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)
            dataloader_list.append(dataloader)

        return dataloader_list
