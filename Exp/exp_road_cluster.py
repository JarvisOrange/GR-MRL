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

    if not os.path.exists(model_path):
        logger.info('please pretrain time patch encoder first.')
        return 
    
    provider = RoadDataProvider(cfg, flag='road_cluster', logger=logger)
    dataloader_list = provider.generate_road_cluster_dataloader()

    model = TSFormer(cfg['TSFormer']).to(device)
    model.mode = 'test'

    model.load_state_dict(torch.load(model_path))

    K = cfg['road_cluster_k']
    dim_embed = cfg['TSFormer']['out_dim']

    road_pattern = torch.zeros([K, dim_embed]).float().cpu()

    for k, dataloader in enumerate(dataloader_list):
        num_embed = dataloader.dataset.get_x_num()

        embed_pool = torch.zeros([num_embed, dim_embed]).float()

        counter = 0
        for batch in dataloader:

            x = batch.permute(0,2,1) # B l_his 7 - > B 7 l_his
            
            H = model(x)

            B,  D = H.shape

            embed_pool[counter : B + counter,:] = H.detach()

            counter = B + counter

        embed = embed_pool.mean(dim=0)
        road_pattern[k,:] = embed.cpu() 

    save_dir = './Save/road_pattern/{}/'.format(dataset_src)
    ensure_dir(save_dir)
    
    save_path = save_dir + 'embed_src_{}.pt'.format(str(K))

    torch.save(road_pattern, save_path)
    logger.info('Road pattern Saved at {}'.format(save_path))

    
    
    