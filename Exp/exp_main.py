import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from RAG_Component.databases import Vectorbase
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from RAG_Component.databases import *


def generate_prompt():
    pass


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
    model = GR_MRL()

    temp, _= cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    embed_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)
    time_embed_pool = torch.load(embed_path).to(device)
    vd = VectorDatabase(time_embed_pool)

    # init exp settings


    #source train
    
    for batch in source_dataloader:
        pass


    #target train
    for batch in target_dataloader:
        pass


    #test train
    for batch in target_dataloader:
        pass


        

   