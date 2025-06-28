import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import os
import torch
from GR_MRL import GR_MRL
from utils import kmeans_pytorch
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
import copy




def exp_time_cluster(cfg, logger=None):
    debug = cfg['debug']
    
    device = cfg['device']

    temp, _ = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    save_dir = './Save/time_embed/{}/'.format(dataset_src)

    embed_path = save_dir + 'embed_src.pt'.format(dataset_src)

    if not os.path.exists(embed_path):

        model_path = './Save/pretrain_model/{}/best_model.pt'.format(dataset_src)
        if not os.path.exists(model_path):
            logger.info(':( please pretrain time patch encoder first.')
            return 
        
        model = TSFormer(cfg['TSFormer']).to(device)
        model.load_state_dict(torch.load(model_path))
        model.mode = 'test'


        provider = RoadDataProvider(cfg, flag='time_cluster', logger=logger)
        dataloader = provider.generate_dataloader()


        num_embed = dataloader.dataset.get_x_num()
        dim_embed = cfg['TSFormer']['out_dim']

        time_embed_pool = torch.zeros([num_embed, dim_embed]).float()

        counter = 0
        for batch in dataloader:

            x = batch.permute(0,2,1) # B l_his 7 - > B 7 l_his
            
            H = model(x)

            B, L, D = H.shape # B * L, D

            H = H.reshape(B * L ,D)

            time_embed_pool[counter:counter+B, :] = H.detach().cpu()

            counter += B 
        
        logger.info("{} emb_pool shape : {}".format(dataset_src, time_embed_pool.shape))

        time_embed_pool.requires_grad = False


        
        ensure_dir(save_dir)

        torch.save(time_embed_pool, save_dir + 'embed_src.pt')

        logger.info('Time Embed Saved at {}'.format(save_dir + 'embed_src.pt'))        


    else:
        time_embed_pool = torch.load(embed_path).to(device)
        time_embed_pool.requires_grad = False

    
    center_label, center, metirc = kmeans_pytorch(time_embed_pool , cfg['time_cluster_k'], device=device)
    
    #s-score [-1,1], 1 is best
    #ch-score, bigger is better
    #db-score, smaller is better
    ch_score, db_score = metirc

    logger.info(f"$$$ Time Pattern Kmeans Metrics: ch-score: {ch_score:.3f}, db-score: {db_score: .3f}")

    save_dir = './Save/time_pattern/{}/'.format(dataset_src)

    ensure_dir(save_dir)

    pattern_path = save_dir + 'pattern_{}.pt'.format(cfg['time_cluster_k'])

    cluster_label_path = save_dir + 'pattern_label_{}.pt'.format(cfg['time_cluster_k'])

    torch.save(center, pattern_path)
    torch.save(center_label,  cluster_label_path)

    logger.info('Time pattern Saved at {}'.format(pattern_path))
    logger.info('Time pattern cluster label Saved at {}'.format(cluster_label_path))

    