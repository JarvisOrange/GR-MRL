import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from Data.prompt_dataset import Vectorbase
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from Data.prompt_dataset import *
import faiss


def exp_main(logger=None):
    device = cfg['device']

    temp, _= cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    source_provider = RoadDataProvider(cfg, flag='source_train', logger=logger)
    source_dataloader = source_provider.generate_dataloader()

    target_provider = RoadDataProvider(cfg, flag='target_train', logger=logger)
    target_dataloader = target_provider.generate_dataloader()

    test_provider = RoadDataProvider(cfg, flag='target_train', logger=logger)
    test_dataloader = test_provider.generate_dataloader()


    # init model and vectorbase
    model = GR_MRL('train')

    

    # init exp settings


    # source train
    for epoch in cfg['flag']['source_train']:
        for batch in source_dataloader:
            output = model(batch)


    # target finetune
    for epoch in cfg['flag']['target_train']:
        for batch in target_dataloader:
            pass


    # inference
    model = GR_MRL('test')
    for batch in test_dataloader:
        pass


        

   