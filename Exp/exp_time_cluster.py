import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from utils import kmeans_pytorch
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
import copy


def exp_time_cluster(dataset_src, logger=None):
    device = cfg['device']

    temp, _ = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    embed_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)

    if os.exist(embed_path):
        time_embed_pool = torch.load(embed_path).to(device)
        time_embed_pool.requires_grad = False

        center, center_label = kmeans_pytorch(
        X = time_embed_pool , num_clusters=cfg['time_cluster_k'])
    
        torch.save(center,'./Save/time_pattern/{}/{}_c.pt'.format(dataset_src, cfg['time_cluster_k']))
        torch.save(center_label,'./Save/time_pattern/{}/{}_cl.pt'.format(dataset_src, cfg['time_cluster_k']))

    else:
        model_path = './Save/pretrain_model/{}/best_model.pt'.format(dataset_src)
        if not os.exist(model_path):
            logger.info('please pretrain time patch encoder first.')
            return 
        
        model = TSFormer(cfg['TSFromer']).to(device)
        model.mode = 'test'
        model.load_state_dict(torch.load(model_path))

        provider = RoadDataProvider(cfg, flag='time_cluster', logger=logger)
        dataloader = provider.generate_dataloader()


        num_embed = dataloader.dataset.get_x_num()
        dim_embed = cfg['TSFormer']['out_channel']

        time_embed_pool  = torch.tensor([num_embed, dim_embed]).float()

        temp = 0
        for batch in tqdm(dataloader):

            x = batch.permute(0,2,1) # B l_his 7 - > B 7 l_his
            
            H = model(x)

            B,  C, L = x.shape

            H = H.reshape(B *  L, -1) # N_time_patch * dim

            logger.info('corresbonding H shape : {}'.format(H.shape))

            time_embed_pool[temp : x.shape[0] + temp,:] = H.detach().cpu()

            temp = x.shape[0] + temp
        
        logger.info("{} emb_pool shape : {}".format(dataset_src, time_embed_pool.shape))

        time_embed_pool.requires_grad = False

        torch.save(time_embed_pool,'./Save/time_embed/{}/embed_src.pt'.format(dataset_src))

        center, center_label = kmeans_pytorch(
        X=time_embed_pool , num_clusters=cfg['time_cluster_k'], device=device)
    
        torch.save(center, './Save/time_pattern/{}/embed_{}.pt'.format(dataset_src, cfg['time_cluster_k']))
        torch.save(center_label, './Save/time_pattern/{}/{}_cl.pt'.format(dataset_src, cfg['time_cluster_k']))