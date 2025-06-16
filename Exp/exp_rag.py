import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from utils import kmeans_pytorch
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from Data.prompt_dataset import *


def generate_related():


def generate_prompt():

def exp_rag(logger=None):

    generate_related()
    

    generate_prompt()

    device = cfg['device']

    temp, _= cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    provider = RoadDataProvider(cfg, flag='rag', logger=logger)
    dataloader = provider.generate_dataloader()

    embed_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)
    info_path = './Save/dataset_info/{}/info.json'.format(dataset_src)

    if os.exist(embed_path):
        time_embed_pool = torch.load(embed_path).to(device)
        time_embed_pool.requires_grad = False
    else:
        logger.info('please do time cluster first')

    with open(info_path, 'r') as f:
        dataset_info_dict = json.load(f)

    dataset_info_list = [dataset_info_dict[k] for k in dataset_info_dict.keys()]
    length = len(dataset_info_dict)

    road_num_list = [dataset_info_list[i]['road_num'] for i in range(length)]
    start_num_list = [dataset_info_list[i]['start_num'] for i in range(length)]
    end_num_list = [dataset_info_list[i]['end_num'] for i in range(length)]
    neighbor_list = [dataset_info_list[i]['adj'] for i in range(length)]

    database = VectorDataset(time_embed_pool)

    # two points split three city data
    index_split_based_city = \
        [0, start_num_list[1], start_num_list[2]]

    related_dict = {}

    
    for index in range(time_embed_pool.shape[0]):
        related_dict = {}
        
        if index < index_split_based_city[1]: # city 0
            city_flag = 0
        elif index < index_split_based_city[2]: # city 1
            city_flag = 1
        else:                                  # city 2
            city_flag = 2

        road_num = road_num_list[city_flag]
        start_num = start_num_list[city_flag]
        end_num = end_num_list[city_flag]
        neighbor_dict = neighbor_list[city_flag]

        # next time step's self feature
        if index + road_num >= end_num: 
            continue

        time_related_index = [index + road_num] 

        # next time step's neighbor  
        road_id = (index - start_num) % road_num
        
        spatial_related_index = \
            [index + (neighbor - road_id) + road_num  \
                for neighbor in neighbor_dict[road_id] \
                if index + (neighbor - road_id) + road_num < end_num] 
        result += time_related_index + spatial_related_index

        related_dict[index] = result

    path = './Save/road_related/{}/result.json'.format(dataset_src)

    with open(path, 'w') as f1:
        json.dump(related_dict, f, indent=4)
        

    


        

   