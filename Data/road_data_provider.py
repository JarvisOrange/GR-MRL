from torch.utils.data import Data, Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import json
from utils import *

from road_data import RoadData


# pretrain dataset(mask) flag pretrain
# cluster dataset(all) flag cluster
# cluster dataset(traffic) flag cluster
# rag_dataset flag rag
# finetune dataset flag finetune
# finetune dataset(test)
# try flag try


class RoadDataset(Dataset):
    def __init__(self, flag, X, Y=None, device=None):
        super(RoadDataset,self).__init__()
        self.X = X
        self.Y = Y
        self.flag = flag

        self.device = device


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.flag == 'pretrain':
            x= self.X[index]
            x = torch.from_numpy(x).float()
            return x

        if self.flag == 'time_cluster' or \
            self.flag == 'road_cluster' or \
            self.flag == 'road_cluster' or \
            self.flag == 'rag':

            x= self.X[index]
            x = torch.from_numpy(x).float()
            x.requires_grad = False
            return x

        if self.flag =='source_train' or \
            self.flag == 'target_train' or \
            self.flag == 'test':
             
            x = self.X[index]
            y = self.Y[index]
            x.requires_grad = False
            y.requires_grad = False
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).float().to(self.device)
            return x, y
            

        if self.flag =='try':
            pass
        
    def get_x_num(self):
        return self.X.shape[0]
    


class RoadDataProvider():
    def __init__(self, cfg, flag, logger=None):
        super(RoadDataProvider, self).__init__()
        
        self.flag = flag
        self,cfg = cfg
        

        self.source_data, self.target_data = cfg['target_data'].split('_')
        self.source_data = self.source_data.split('-')

        self.target_data = self.target_data
        self.source_data = [n for n in self.source_data]

        if flag == 'source_train' or flag == 'target_train' or flag == 'test':
            self.data_list = [RoadData(cfg, self.source_data[i], flag=self.flag, logger=logger) for i in range(3)]

            X_num = 0, X_num_dict = {}
        
            his_num, pre_num, interval = self.data_list[0].get_data_info()
            
            for dataset in self.data_src_list:
                temp = dataset.get_data_num()
                X_num += temp
                X_num_dict[dataset.name] = temp
                
            self.X = np.zeros([X_num, his_num, 7], dtype=float)
            self.Y = np.zeros([X_num, pre_num, 7], dtype=float)

            cur = 0
            for dataset in self.data_list:
                x, y = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :, :] = x
                self.Y[cur:cur + X_num_dict[dataset.name], :, :] = y

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
                
            self.X = np.zeros([X_num, his_num, 7], dtype=float)

            cur = 0
            for dataset in self.data_src_list:
                x = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :, :] = x

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
                
            self.X = np.zeros([X_num, his_num, 7], dtype=float)

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

            cluster_label,_ = kmeans_pytorch(self.Road_E, self.cfg['road_k']) #[label_2, label_10..., label_1]

            self.road_cluster_label = cluster_label

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
                self.X_road_cluster_dict[k] = temp_x #[, S*l_his, 7]
            

        if flag == 'rag':
            ### time cluster task have get the dataset and transfer into embed
            ### this dataset is to get the dataset info and save into json
            self.data_time_cluster_list = [RoadData(cfg, self.source_data[i], flag='rag', logger=logger) for i in range(3)]

            dataset_info_dict = {}
        
            his_num, pre_num, interval = self.data_src_list[0].get_data_info()

            cur = 0
            for dataset in self.data_src_list:
                data_num = dataset.get_data_num()
                road_num = dataset.get_road_num()

                dataset_info_dict['name'] = {}
                dataset_info_dict['name']['data_num'] = data_num * road_num
                dataset_info_dict['name']['road_num'] = road_num
                dataset_info_dict['name']['data_start_num'] = cur
                dataset_info_dict['name']['data_end_num'] = cur + data_num * road_num
                dataset_info_dict['name']['adj'] = adj_to_dict(dataset.get_adj()) 

                cur += cur + data_num * road_num

            json_path = '/Save/dataset_info/{}/info.json'.format(self.source_data)
            with open(json_path, 'w') as f:
                json.dump(dataset_info_dict, f, indent=4)
        

        if flag =='try':
            pass
    

    def generate_dataloader(self):
        drop_last = False
        shuffle = False
        
        if self.flag == 'pretrain': 
            bs = self.cfg['flag'][self.flag]['batch_size']
        else: # source_train target_train test rag time_cluster road_cluster don't need shuffle
            bs = 1

        R_dataset = RoadDataset(self.flag, self.X, self.Y, device = self.cfg['device'])

        dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = shuffle, drop_last=drop_last)

        return dataloader
    
    
    def generate_pretrain_dataloader(self):
        bs = self.cfg['flag']['self.flag']['batch_size']
        drop_last = self.cfg['drop_last']

        train_ratio, val_ratio, test_ratio = self.cfg['flag']['pretrain']['train_val_test']

        length = self.X.shape[0]
        X_train = self.X[ : int(train_ratio*length)]
        X_val = self.X[int( train_ratio*length) : int((train_ratio+val_ratio)*length)]
        X_test = self.X[int((1-test_ratio)*length) : ]
        

        indices = np.random.permutation(self.X_train.shape[0])
        X_train_shuffle = X_train[indices]

        indices = np.random.permutation(self.X_val.shape[0])
        X_val_shuffle = X_val[indices]

        indices = np.random.permutation(self.X_test.shape[0])
        X_test_shuffle = X_test[indices]

        R_train_dataset = RoadDataset(self.flag, X_train_shuffle, device = self.cfg['device'])
        R_val_dataset = RoadDataset(self.flag, X_val_shuffle, device = self.cfg['device'])
        R_test_dataset = RoadDataset(self.flag, X_test_shuffle, device = self.cfg['device'])

        train_dataloader = DataLoader(R_train_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)

        val_dataloader = DataLoader(R_val_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)

        test_dataloader = DataLoader(R_test_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)

        return train_dataloader, val_dataloader, test_dataloader

    def generate_road_cluster_dataloader(self):
        
        assert self.flag == 'road_cluster', 'this provider is not for road clustering'
        bs = self.cfg['pretrain']['batch_size']
        drop_last = False

        dataloader_list = []

        for k in self.X_road_cluster_dict.keys():
            R_dataset = RoadDataset(self.flag, self.X_road_cluster_dict[k], device = self.cfg['device'])
            dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)
            dataloader_list.append(dataloader)

        return dataloader_list


    def get_road_cluster_label(self):
        return self.road_cluster_label