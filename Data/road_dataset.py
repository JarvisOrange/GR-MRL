from torch.utils.data import Dataset, DataLoader
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
        
        self.dim = 128

        self.source_data, self.target_data = cfg['target_data'].split('_')
        self.source_data = self.source_data.split('-')

        self.target_data = dataset_name_dict[self.target_data]
        self.source_data = [dataset_name_dict[n] for n in self.source_data]

        if flag == 'finetune':
            self.data_src_1 = RoadData(cfg, self.source_data[0], stage='source_train', logger=logger)
            self.data_src_2 = RoadData(cfg, self.source_data[1], stage='source_train', logger=logger)
            self.data_src_3 = RoadData(cfg, self.source_data[2], stage='source_train', logger=logger)

            self.data_tgt = RoadData(cfg, self.target_data, stage='target', logger=logger)

        if flag == 'cluster':
            pass

        if flag == 'pretrain':
            pass

        if flag == 'rag':
            pass

        if flag =='try':
            pass


    def combine_node_embed(self, ):
        laplace_e = self.get_laplace_matrix()
        spacesyntax_e = self.get_space_syntax_matrix()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


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
