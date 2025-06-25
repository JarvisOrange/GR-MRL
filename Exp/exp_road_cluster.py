import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import os
import torch
from GR_MRL import GR_MRL

from utils import *
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *


def exp_road_cluster(cfg, logger=None):
    debug  = cfg['debug']

    device = cfg['device']

    temp, _ = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))
    model_path = './Save/pretrain_model/{}/best_model.pt'.format(dataset_src)

    if not os.exist(model_path):
        logger.info('please pretrain time patch encoder first.')
        return 
    
    provider = RoadDataProvider(cfg, flag='road_cluster', logger=logger)
    dataloader_list = provider.generate_road_cluster_dataloader()

    model = TSFormer(cfg['TSFromer']).to(device)
    model.mode = 'test'

    model.load_state_dict(torch.load(model_path))

    K = cfg['road_cluster_k']
    dim_embed = cfg['TSFormer']['out_dim']

    road_pattern = torch.zeros([K, dim_embed]).float().cpu()

    for k, dataloader in enumerate(dataloader_list):
        num_embed = dataloader.dataset.get_x_num()

        embed_pool = torch.zeros([num_embed, dim_embed]).float().cpu()

        temp = 0
        for batch in tqdm(dataloader):

            x = batch.permute(0,2,1) # B l_his 7 - > B 7 l_his
            
            H = model(x)

            B,  C, L = x.shape

            H = H.reshape(B * L, -1) # N_time_patch * dim

            logger.info('corresbonding H shape : {}'.format(H.shape))

            embed_pool[temp : x.shape[0] + temp,:] = H.detach().cpu()

            temp = x.shape[0] + temp

        embed = embed_pool.mean(dim=0)
        road_pattern[k,:] = embed 
    

    torch.save(road_pattern,'./Save/road_pattern/{}/embed_{}.pt'.format(dataset_src, str(K)))

    
    
    