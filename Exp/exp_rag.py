import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from utils import kmeans_pytorch
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from Data.prompt_dataset import *


dataset_description = {
    0: "The PMES-BAY dataset comprises traffic speed data collected from 325 traffic sensors placed on San Francisco Bay Area freeway system."
        "The data was recorded at five-minute intervals from Jan 1st to June 30th, 2017.",

    1: "The METR-LA dataset consists of traffic speed data collected from 207 loop sensors placed on freeway sections within Los Angeles County."
        "The data was recorded at five-minute intervals from May 1st to June 30th, 2012.",

    2: "The Chengdu dataset comprises traffic speed data collected from 524 road links in Chengdu, China."
        "The data was recorded at five-minute intervals from Jan 1st to Apr 30th, 2018.",

    3:"The Shenzhen dataset comprises traffic speed data collected from 627 road links in Shenzhen, China."
        "The data was recorded at five-minute intervals from Jan 1st to Apr 30th, 2018."
}


def generate_related(dataset_src, time_embed_pool, logger):

    info_path = './Save/dataset_info/{}/info.json'.format(dataset_src)

    with open(info_path, 'r') as f:
        dataset_info_dict = json.load(f)

    dataset_info_list = [dataset_info_dict[k] for k in dataset_info_dict.keys()]
    length = len(dataset_info_dict)

    road_num_list = [dataset_info_list[i]['road_num'] for i in range(length)]
    start_num_list = [dataset_info_list[i]['start_num'] for i in range(length)]
    end_num_list = [dataset_info_list[i]['end_num'] for i in range(length)]
    neighbor_list = [dataset_info_list[i]['adj'] for i in range(length)]

    

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
        json.dump(related_dict, f1, indent=4)


def generate_prompt(self, dataset_src, dataloader, logger):
        model_path = './Save/pretrain_model/{}/best_model.pt'.format(dataset_src)
        if not os.exist(model_path):
            logger.info('please pretrain time patch encoder first.')
            return 
        
        model = TSFormer(cfg['TSFromer']).to(cfg['device'])
        model.mode = 'test'
        model.load_state_dict(torch.load(model_path))

        prompt_list = {}

        k = cfg['retrieve_k']

        for index, batch in enumerate(dataloader):
            x, y = batch
            city = batch[0, 0 ,4]
            means = batch[0, 0, 5] 

            x = batch.permute(0, 2, 1) # 1 l_his 7 - > 1 7 l_his
            
            H = model(x)

            B, C, L = x.shape

            H = H.reshape(B *  L, -1) # N_time_patch,  dim
            
            related_list = self.query_related(index, self.vectors[index], k)

            prompt_base = "Dataset Description: {}".format(dataset_description[city]) + \
                 f"Dataset statisctis: The mean speed value of the dataset is {str(means)}" + \
                 f"Task Description: Predict the next {str(cfg['pre_num'])} steps given the history data and corresponding tokens. " + \
                 "Reference tokens can also be useful for prediction."
                
            
        prompt_list[index] = {'index': index, 'ref': related_list, 'prompt': prompt_base, 'truth': y}

        return prompt_list
            

def exp_rag(logger=None):

    device = cfg['device']

    temp, trg = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    embed_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)
    

    if os.exist(embed_path):
        time_embed_pool = torch.load(embed_path).to(device)
        time_embed_pool.requires_grad = False
    else:
        logger.info('please do time cluster first')
        return

    related_path = './Save/road_related/{}/result.json'.format(dataset_src)

    if not os.exist(related_path):
        logger.info('generate related road first')
        generate_related(dataset_src, time_embed_pool, logger)

    database = VectorDataset(dataset_src, time_embed_pool)

    source_provider = RoadDataProvider(cfg, flag='source_train', logger=logger)
    source_dataloader = source_provider.generate_dataloader()
    src_prompt = generate_prompt(dataset_src, source_dataloader, logger)
    src_json_path = 'Save/prompt/{}/src.json'.format(dataset_src)
    with open(src_json_path, 'w') as f:
                json.dump(src_prompt , f, indent=4)


    target_provider = RoadDataProvider(cfg, flag='target_train', logger=logger)
    target_dataloader = target_provider.generate_dataloader()
    trg_prompt = generate_prompt(dataset_src, target_dataloader)
    trg_json_path = 'Save/prompt/{}/trg.json'.format(dataset_src)
    with open(src_json_path, 'w') as f:
                json.dump(trg_prompt , f, indent=4)

    test_provider = RoadDataProvider(cfg, flag='test', logger=logger)
    test_dataloader = test_provider.generate_dataloader()
    test_prompt = generate_prompt(dataset_src, test_dataloader)
    test_json_path = 'Save/prompt/{}/test.json'.format(dataset_src)
    with open(src_json_path, 'w') as f:
                json.dump(trg_prompt , f, indent=4)
    

    
        

    


        

   