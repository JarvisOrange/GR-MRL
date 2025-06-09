import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from utils import kmeans_pytorch
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from RAG_Component.databases import *




def exp_rag(dataset_src, logger=None):

    # may retrieve other city time patch
    def get_related_index(retrieve_result, dataset_info_list, index_split_based_city):
        

        return result


    device = cfg['device']

    temp, _= cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    provider = RoadDataProvider(cfg, flag='rag', logger=logger)
    dataloader = provider.generate_dataloader()

    embed_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)
    info_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)

    if os.exist(embed_path):
        time_embed_pool = torch.load(embed_path).to(device)
        with open(info_path, 'r') as f:
            dataset_info_dict = json.load(f)

    dataset_info_list = [dataset_info_dict[k] for k in dataset_info_dict.keys()]
    length = len(dataset_info_dict)

    road_num_list = [dataset_info_list[i]['road_num'] for i in range(length)]
    start_num_list = [dataset_info_list[i]['start_num'] for i in range(length)]
    end_num_list = [dataset_info_list[i]['end_num'] for i in range(length)]
    neighbor_list = [dataset_info_list[i]['adj'] for i in range(length)]

    database = VectorDatabase(time_embed_pool)

    source_datasets = dataset_src.split('-')

    # two points split three city data
    index_split_based_city = \
        [0, start_num_list[1], start_num_list[2]]

    
    retrieve_dict = {}

    top_k = cfg['retrieve_k']
    
    for index in range(time_embed_pool.shape[0]):
        v = time_embed_pool[index, :]

        result = database.query(v, top_k)

        result = [x for x in result if x != index] # delete self

        retrieve_dict[index] = get_related_index(result, dataset_info_list, index_split_based_city)

        retrieve_result = []
        for res in result:
            if res < index_split_based_city[1]: # city 0
                city_flag = 0
            elif res < index_split_based_city[2]: # city 1
                city_flag = 1
            else:                                  # city 2
                city_flag = 2

            road_num = road_num_list[city_flag]
            start_num = start_num_list[city_flag]
            end_num = end_num_list[city_flag]
            neighbor_dict = neighbor_list[city_flag]

            # next time step's self feature
            if res + road_num >= end_num: 
                continue

            time_related_index = [res + road_num] 

            # next time step's neighbor  
            road_id = (res - start_num) % road_num
            
            spatial_related_index = \
                [res + (neighbor - road_id) + road_num  \
                 for neighbor in neighbor_dict[road_id] \
                 if res + (neighbor - road_id) + road_num < end_num] 
        result += time_related_index + spatial_related_index

    path = './Save/retrieve_Result/{}/result.json'.format(dataset_src)

    with open(path, 'w') as f1:
        json.dump(retrieve_dict, f, indent=4)
        

    


        

   