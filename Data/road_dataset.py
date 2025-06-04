from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from utils import *

from Node_Embed_Model.Laplacian import *
from Node_Embed_Model.SpaceSyntax import *
from road_data import RoadData
# pretrain dataset(mask)
# cluster dataset(all)
# cluster dataset(traffic)
# rag_dataset
# finetune dataset
# finetune dataset(test)
# try


class RoadDataset(Dataset):
    def __init__(self, cfg, stage, logger=None):
        super(RoadDataset, self).__init__()
        self.data = RoadData(cfg, stage, logger)
        self.dim = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_laplace_matrix(self):
        self.adj = self.dataset.get_adj()
        return get_laplac_embed(self.adj)

    def get_space_syntax_matrix(self):
        self.adj = self.dataset.get_adj()
        return get_space_syntax_embed(self.adj)

def data_provide(cfg, stage):
    Data = dataset_dict[cfg['dataset_source']]

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

    laplace_embed = data_set.get_laplace_matrix()
    deep_walk_embed = data_set.get_deep_walk_matrix()
    space_syntax_embed = data_set.get_space_syntax_matrix()

    a, b, c = data_set.get_dataloader()

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=args.num_workers, # not very stable
        drop_last=drop_last,
    )

    return data_set, data_loader
