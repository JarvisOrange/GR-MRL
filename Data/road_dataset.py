from torch.utils.data import Data, Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from utils import *

from Node_Embed_Model.Laplacian import *
from Node_Embed_Model.SpaceSyntax import *
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
    def __init__(self, cfg, flag, logger=None):
        super(RoadDataset, self).__init__()
        
        self.flag = flag
        
        self.dim = 128

        self.source_data, self.target_data = cfg['target_data'].split('_')
        self.source_data = self.source_data.split('-')

        self.target_data = dataset_name_dict[self.target_data]
        self.source_data = [dataset_name_dict[n] for n in self.source_data]

        if flag == 'finetune':
            self.data_src_list = [RoadData(cfg, self.source_data[i], stage='source_train', logger=logger) for i in range(3)]
            
            self.data_tgt = RoadData(cfg, self.target_data, stage='target', logger=logger)

            self.data_test = RoadData(cfg, self.target_data, stage='test', logger=logger)

        if flag == 'cluster':
            self.data_time_cluster_list = [RoadData(cfg, self.source_data[i], stage='time_cluster', logger=logger) for i in range(3)]
        
            self.data_road_cluster_list = [RoadData(cfg, self.source_data[i], stage='road_cluster', logger=logger) for i in range(3)]
            

        if flag == 'pretrain':
            self.data_src_list = [RoadData(cfg, self.source_data[i], stage='pretrain', logger=logger) for i in range(3)]
            
            self.pretrain_batch_num = 0
            batch_size = cfg['pretrain']['batch_size']
            self.batchnum_dict = {}
            
            for dataset in self.data_src_list:
                x, y = dataset.get_data()

                batch_num = int(x // batch_size)
                self.batchnum_dict[dataset.name] = batch_num
                self.pretrain_batch_num += batch_num 
            
            self.pretrain_which_data = torch.zeros((self.pretrain_batch_num))
            self.pretrain_which_pos = torch.zeros((self.pretrain_batch_num))

            cur = 0
            for idx, dataset in enumerate(self.data_src_list):
                self.pretrain_which_data[cur : cur + self.batch_num_dict[dataset.name]] = int(idx)
                self.pretrain_which_pos[cur : cur + self.batch_num_dict[dataset.name]] = torch.arange(cur, cur + self.batch_num_list[dataset.name]) - cur
                cur += self.batch_num_list[dataset.name]
            self.random_permutation =torch.randperm(self.pretrain_batch_num)


        if flag == 'rag':
            pass

        if flag =='try':
            pass


    def combine_node_embed(self, ):
        laplace_e = self.get_laplace_matrix()
        spacesyntax_e = self.get_space_syntax_matrix()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if self.flag == 'finetune':
            pass

        if self.flag == 'cluster':
            pass
            

        if self.flag == 'pretrain':
            # need query *batch_size* continuous batches
            idx = self.random_permutation[index]
            select_dataset = self.data_list[self.pretrain_which_data[idx].detach().cpu().numpy().astype(int)]
            pos = self.pretrain_which_pos[idx].detach().cpu().numpy().astype(int)
            batch_size = self.task_args['batch_size']
            
            indices = torch.tensor(list(range(pos,pos+batch_size)))
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        if flag == 'rag':
            pass

        if flag =='try':
            pass

        x_data = x_data.float()
        y_data = y_data.float()
        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data,means=self.means_list[select_dataset],stds = self.stds_list[select_dataset])
        data_i.data_name = select_dataset
        
        return data_i


    def get_laplace_matrix(self):
        adj = self.dataset.get_adj()
        return get_laplac_embed(self.adj)


    def get_space_syntax_matrix(self):
        adj = self.dataset.get_adj()
        return get_space_syntax_embed
    

    def data_provide(cfg, stage):
        if stage == "try":
            flag_try=False
            shuffle = False
            drop_last = True
            batch_size = 1  # bsz=1 for evaluation
        else:
            flag_try=True
            shuffle = True
            drop_last = True
            batch_size = cfg['batch_size']  # bsz for train and valid

    def generate_dataloader(self):
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=args.num_workers, # not very stable
            drop_last=drop_last,
        )

        return data_set, data_loader
