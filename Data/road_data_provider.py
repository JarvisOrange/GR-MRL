
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import json
from utils import *

from .road_data import RoadData


# pretrain dataset(mask) flag pretrain
# cluster dataset(all) flag cluster
# cluster dataset(traffic) flag cluster
# rag_dataset flag rag
# finetune dataset flag finetune
# finetune dataset(test)


class RoadDataset(Dataset):
    def __init__(self, flag, X, Y, device=None):
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
            x = torch.from_numpy(x).float().to(self.device)
            x.requires_grad = False
            return x

        if self.flag =='source_train' or \
            self.flag == 'target_train' or \
            self.flag == 'test':
             
            x = self.X[index]
            y = self.Y[index]
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).float().to(self.device)
            x.requires_grad = False
            y.requires_grad = False
            return x, y
            

    def get_x_num(self):
        return self.X.shape[0]
    


class RoadDataProvider():
    def __init__(self, cfg, flag, logger=None):
        
        super(RoadDataProvider, self).__init__()
        
        self.flag = flag
        
        self.cfg = cfg
        
        self.device = cfg['device']
        

        self.source_data, self.target_data = cfg['dataset_src_trg'].split('_')
        self.source_data = self.source_data.split('-')

        self.source_data = [n for n in self.source_data]
        

        # if flag == 'pretrain':
    
        #     self.data_src_list = [
        #         RoadData(cfg, self.source_data[i], flag, logger=logger) 
        #         for i in range(3)]

        #     X_num, X_train_num, X_val_num = 0, 0, 0
        #     X_train_num_dict, X_val_num_dict = {}, {}
        #     X_num_dict = {}
        
        #     his_num, pre_num, interval = self.data_src_list[0].get_data_info()
        #     train_ratio, val_ratio = self.cfg['flag']['pretrain']['train_val_test']
        #     for dataset in self.data_src_list:
        #         temp = dataset.get_x_num()
        #         X_num += temp
        #         X_num_dict[dataset.name] = temp

        #         X_train_num += int(temp * train_ratio)
        #         X_val_num += temp - int(temp * train_ratio)
        #         X_train_num_dict[dataset.name] = int(temp * train_ratio)
        #         X_val_num_dict[dataset.name] = temp - int(temp * train_ratio)
                
        #     self.X = np.zeros([X_num, his_num, 7], dtype=float)
        #     self.Y = None

        #     self.X_train = np.zeros([X_train_num, his_num, 7], dtype=float)
        #     self.X_val = np.zeros([X_val_num, his_num, 7], dtype=float)

        #     cur = 0
        #     for dataset in self.data_src_list:
        #         x = dataset.get_data()
        #         self.X[cur:cur + X_num_dict[dataset.name], :, :] = x
        #         cur += dataset.get_x_num()
        #     cur1, cur2 = 0, 0
        #     for dataset in self.data_src_list:
        #         x = dataset.get_data()
                
        #         x_train = x[:X_train_num_dict[dataset.name],:,:]
        #         x_val = x[X_train_num_dict[dataset.name]:,:,:]

        #         self.X_train[cur1:cur1 + X_train_num_dict[dataset.name], :, :] = x_train
        #         self.X_val[cur2:cur2 + X_val_num_dict[dataset.name], :, :] = x_val

        #         cur1 += X_train_num_dict[dataset.name]
        #         cur2 += X_val_num_dict[dataset.name]

        if flag == 'time_cluster':
            # time patch
            self.data_time_cluster_list = [
                RoadData(cfg, self.source_data[i], flag='time_cluster', logger=logger) 
                for i in range(3)]

            X_num = 0
            X_num_dict = {}
        
            his_num, pre_num, interval = self.data_time_cluster_list[0].get_data_info()
        
        if flag == 'road_cluster':
            # road
            self.data_road_cluster_0 = RoadData(cfg, self.source_data[0], flag='road_cluster', logger=logger)
            self.data_road_cluster_1 = RoadData(cfg, self.source_data[1], flag='road_cluster', logger=logger)
            self.data_road_cluster_2 = RoadData(cfg, self.source_data[2], flag='road_cluster', logger=logger)
            self.data_road_cluster_list = [self.data_road_cluster_0, self.data_road_cluster_1, self.data_road_cluster_2]
            
            R_num = 0
            R_num_dict = {}

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

            self.Road_E = torch.from_numpy(self.Road_E).float()
            cluster_label, _, metrics = kmeans_pytorch(self.Road_E, self.cfg['road_cluster_k']) #[label_2, label_10..., label_1]
            ch_score, db_score = metrics
            logger.info(f"$$$ Road Kmeans Metrics: ch-score: {ch_score:.3f}, db-score: {db_score: .3f}")
            self.road_cluster_label = cluster_label.numpy()

            cluster_label_node_dict = {}
            self.X_road_cluster_dict = {}

            self.X_road_cluster_0 = self.data_road_cluster_0.get_data() # [S, N, 2, l_his]
            self.X_road_cluster_1 = self.data_road_cluster_1.get_data() # [S, N, 2, l_his]
            self.X_road_cluster_2 = self.data_road_cluster_2.get_data() # [S, N, 2, l_his]
            
            for k in range(self.cfg['road_cluster_k']):
                cluster_label_node_dict[k] = np.where(cluster_label == k)[0]

            R_num_list = [v for k,v in R_num_dict.items()]

            for k in cluster_label_node_dict.keys():
                temp_list = cluster_label_node_dict[k].tolist()
                temp_x = []
                for i in temp_list:
                    if i < R_num_list[0]:
                        index = i
                        temp_x += [self.X_road_cluster_0[:, index, :, :]]

                    elif i < R_num_list[0] + R_num_list[1]:
                        index = i - R_num_list[0]
                        temp_x += [self.X_road_cluster_1[:, index, :, :]]

                    elif i < R_num_list[0] + R_num_list[1] + R_num_list[2]:
                        index = i - R_num_list[0] - R_num_list[1]
                        temp_x += [self.X_road_cluster_2[:, index, :, :]]
                temp_x = np.vstack(temp_x) # [S1+S2+S3, 1, 2, l_his]
                S, l_fea, l_his = temp_x.shape
                temp_x = temp_x.reshape(-1, 1 , l_fea, l_his)
                self.X_road_cluster_dict[k] = temp_x    
            

        if flag == 'rag':
            # time cluster task have get the dataset and transfer into embed
            # this dataset is to get the dataset info and save into json
            temp, _ = cfg['dataset_src_trg'].split('_')
            dataset_src = ''.join(temp.split('-'))
            save_dir = './Save/dataset_info/{}/'.format(dataset_src)
            ensure_dir(save_dir)

            json_path = save_dir + 'info.json'

            if os.path.exists(json_path): 
                logger.info("{} already exists".format(json_path))

            else:
                self.data_rag_list = [RoadData(cfg, self.source_data[i], flag='rag', logger=logger) for i in range(3)]
                dataset_info_dict = {}
                his_num, pre_num, interval = self.data_rag_list[0].get_data_info()

                cur = 0
                for dataset in self.data_rag_list:
                    name = dataset.name
                    x_num = dataset.get_x_num()
                    road_num = dataset.get_road_num()

                    dataset_info_dict[name] = {}
                    dataset_info_dict[name]['data_num'] = x_num
                    dataset_info_dict[name]['road_num'] = road_num
                    dataset_info_dict[name]['start_num'] = cur
                    dataset_info_dict[name]['end_num'] = cur + x_num
                    dataset_info_dict[name]['adj'] = adj_to_dict(dataset.get_adj()) 

                    cur += cur + x_num * road_num
                    
                with open(json_path, 'w') as f:
                    json.dump(dataset_info_dict, f, indent=4)
                    

        if flag == 'source_train':
            self.data_list = [
                RoadData(cfg, self.source_data[i], flag=self.flag, logger=logger) 
                for i in range(3)]

            X_num = 0
            X_num_dict = {}
        
            his_num, pre_num, interval = self.data_list[0].get_data_info()
            
            for dataset in self.data_list:
                temp = dataset.get_x_num()
                X_num += temp
                X_num_dict[dataset.name] = temp
                
            self.X = np.zeros([X_num, 1, 7, his_num], dtype=float)
            self.Y = np.zeros([X_num, pre_num], dtype=float)

            cur = 0
            for dataset in self.data_list:
                x, y = dataset.get_data()
                
                self.X[cur:cur + X_num_dict[dataset.name], :, :, :] = x
                self.Y[cur:cur + X_num_dict[dataset.name], :] = y

                cur += dataset.get_x_num()

            #target test apply different epoch

        if flag == 'target_train':
            self.data_trg = RoadData(cfg, self.target_data, flag='target_train', logger=logger)

            self.X, self.Y = self.data_trg.get_data()

        if flag == 'test':
            self.data_trg = RoadData(cfg, self.target_data, flag='test', logger=logger)

            self.X, self.Y = self.data_trg.get_data()
            
    def generate_dataloader(self, rag_flag=None):
        #except pretrain and road cluster
        drop_last = False
        shuffle = False

        if rag_flag == 'rag':
            bs = self.cfg['flag']['rag']['batch_size']
        else:
            bs = self.cfg['flag'][self.flag]['batch_size']

        R_dataset = RoadDataset(self.flag, self.X, self.Y, device = self.cfg['device'])
        dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = shuffle, drop_last=drop_last)
        return dataloader
    
    # def generate_pretrain_dataloader(self):
    #     bs = self.cfg['flag'][self.flag]['batch_size']
    #     drop_last = self.cfg['drop_last']
    #     shuffle = self.cfg['flag'][self.flag]['shuffle']
    
    #     train_ratio, val_ratio = self.cfg['flag']['pretrain']['train_val_test']

    #     length = self.X.shape[0]
    #     X_train = self.X_train
    #     X_val = self.X_val
        
    #     if shuffle:
    #         print(shuffle)
    #         indices = np.random.permutation(X_train.shape[0])
    #         X_train= X_train[indices]

    #         indices = np.random.permutation(X_val.shape[0])
    #         X_val = X_val[indices]

    #     R_train_dataset = RoadDataset(self.flag, X_train, Y=None, device = self.cfg['device'])
    #     R_val_dataset = RoadDataset(self.flag, X_val, Y=None, device = self.cfg['device'])

    #     train_dataloader = DataLoader(R_train_dataset, batch_size = bs, drop_last=drop_last)
    #     val_dataloader = DataLoader(R_val_dataset, batch_size = bs, drop_last=drop_last)

    #     return train_dataloader, val_dataloader

    def generate_time_cluster_dataloader(self):
        assert self.flag == 'time_cluster', 'this provider is not for road clustering'
        bs = self.cfg['flag']['road_cluster']['batch_size']
        drop_last = False

        dataloader_list = []
        for d_t in self.data_time_cluster_list:
            X = d_t.get_data()
            Y = None
            R_dataset = RoadDataset(self.flag, X, Y, device = self.cfg['device'])
            dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)
            dataloader_list.append(dataloader)
        return dataloader_list
    

    def generate_road_cluster_dataloader(self):
        assert self.flag == 'road_cluster', 'this provider is not for road clustering'
        bs = self.cfg['flag']['road_cluster']['batch_size']
        drop_last = False

        dataloader_list = []
        for k in self.X_road_cluster_dict.keys():
            Y = None
            R_dataset = RoadDataset(self.flag, self.X_road_cluster_dict[k], Y, device = self.cfg['device'])
            dataloader = DataLoader(R_dataset, batch_size = bs, shuffle = True, drop_last=drop_last)
            dataloader_list.append(dataloader)
        return dataloader_list


    def get_road_cluster_label(self):
        return self.road_cluster_label