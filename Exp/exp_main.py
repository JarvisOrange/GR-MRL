import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from Data.VectorBase import Vectorbase
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from Data.VectorBase import *
from config import cfg



def exp_main(logger=None):
    device = cfg['device']

    temp, _= cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    flag = 'source_train'

    bs = cfg['flag'][flag]['batch_size']

    source_dataset = PromptDataset(dataset_src, 'src')
    source_dataloader = DataLoader(source_dataset, batch_size = bs, shuffle = True, drop_last=False)
    
    for epoch in cfg['flag'][flag]:
        for batch in source_dataloader:
            output = model(batch)


    # init model and vectorbase
    model = GR_MRL('train')

    

    # init exp settings


    # source train
    


    # target finetune
    for epoch in cfg['flag']['target_train']:
        for batch in target_dataloader:
            pass


    # inference
    model = GR_MRL('test')
    for batch in test_dataloader:
        pass


        

   