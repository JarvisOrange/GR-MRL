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
from Data.VectorBase import *



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

    for index in tqdm(range(time_embed_pool.shape[0])):
        
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
        
        # dict from json , keys are str, so it will be turn into int
        neighbor_dict = {int(k): v for k, v in neighbor_dict.items()}

        result = []

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
    
        result = time_related_index + spatial_related_index

        related_dict[index] = list(set(result))


    path = './Save/related/{}/result.json'.format(dataset_src)

    with open(path, 'w') as f1:
        json.dump(related_dict, f1, indent=4)

    logger.info("^.^ generate result.json in {} Success".format(path))


def generate_prompt(cfg, flag, dataset_src, dataloader, vectorbase, logger, dataset_trg=None):
    if flag == 'source_train':
        embed_path = './Save/time_embed/{}/embed_src.pt'.format(dataset_src)
    if flag == 'target_train':
        embed_path = './Save/time_embed/{}/embed_target.pt'.format(dataset_src)
    if flag == 'test':
        embed_path = './Save/time_embed/{}/embed_test.pt'.format(dataset_src)

    if not os.path.exists(embed_path):
        save_flag = 'True'
        num_embed = dataloader.dataset.get_x_num() * int(cfg['his_num'] /cfg['TSFormer']['patch_size'])
        dim_embed = cfg['TSFormer']['out_dim']  # 128
        embed_pool = torch.zeros([num_embed, dim_embed]).cuda().float()
    else:
        save_flag = 'False'
        
    model_path = './Save/pretrain_model/{}/best_model.pt'.format(dataset_src)
    if not os.path.exists(model_path):
        logger.info(':(  please pretrain time patch encoder first.')
        return 
    
    model = TSFormer(cfg['TSFormer']).to(cfg['device'])
    model.load_state_dict(torch.load(model_path))
    model.mode = 'test'
    
    prompt_list = {}
    k = cfg['retrieve_k']
    #####################################################################
    
    cur = 0
    for index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, y = batch

        city = int(x[0, 0, 4].detach().cpu().numpy())
        means = x[0, 0, 5].detach().cpu().numpy()
        stds = x[0, 0, 6].detach().cpu().numpy()
        
        x = x.permute(0, 2, 1) # B l_his 7 - > B 7 l_his

        H = model(x) 
        H = H.detach()

        B, L, D = H.shape # B * L, D

        if save_flag == 'True':
            temp_H = H.reshape(B*L, D)
            embed_pool[cur:cur+B*int(cfg['his_num'] /cfg['TSFormer']['patch_size']), :] = temp_H

        # H (B, 12, D) -> (B, 1, D) calculate B[B, 9:12, D] mean
        H = H[:, -3:, :].mean(dim=1) # B, D
        H = H.reshape(B, D)
            
        related_dict = vectorbase.query_related(H,  k)


        for i in related_dict.keys():
            if flag == 'test':
                prompt_index = cur + i + cfg['target_day']
            else:
                prompt_index = cur + i 
            
            prompt_base = "Dataset Description: {}\n".format(dataset_description[city]) + \
                f"Dataset statisctis: The mean speed value of the dataset is {str(means)} and the std speed value of the dataset is {str(stds)}.\n" + \
                f"Task Description: Given the history data and reference data which may be useful, predict the next {str(cfg['pre_num'])} steps.\n"
            
            if related_dict[i] is not None:
                prompt_list[prompt_index] = {'index': prompt_index, 
                                             'ref': related_dict[i], 
                                             'prompt': prompt_base, 
                                             'label': y[i,:].cpu().numpy().tolist()}
            else:
                prompt_list[prompt_index] = {'index': prompt_index, 
                                             'ref': [], 
                                             'prompt': prompt_base, 
                                             'label': y[i,:].cpu().numpy().tolist()}
                

        cur += B
    if save_flag == 'True':
        torch.save(embed_pool, embed_path)
        logger.info('For ft: {} Embed Saved at {}'.format(flag, embed_path))

    return prompt_list
            

def exp_rag(cfg, logger=None):
    
    debug  = cfg['debug']
    device = cfg['device']

    temp, dataset_trg = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    embed_path = './Save/time_pattern/{}/embed.pt'.format(dataset_src)
    
    if os.path.exists(embed_path):
        time_embed_pool = torch.load(embed_path)
    else:
        logger.info(':( please do time cluster first!')
        return
    
    save_dir = './Save/related/{}/'.format(dataset_src)
    ensure_dir(save_dir)
    related_path = save_dir + 'result.json'.format(dataset_src)

     # rag_provider is to save some improtant json file, not to generate train data
    rag_provider =  RoadDataProvider(cfg, flag='rag', logger=logger)

    if not os.path.exists(related_path):
        logger.info(':( generate related json first!')
        generate_related(dataset_src, time_embed_pool, logger)

    database = VectorBase(cfg, dataset_src, time_embed_pool)

    save_dir = 'Save/prompt/{}/'.format(dataset_src)
    ensure_dir(save_dir)

    logger.info("Start to generate prompt json files for dataset: {}".format(dataset_src))

    flag = 'source_train'
    database.update_mode(flag)
    source_provider = RoadDataProvider(cfg, flag=flag, logger=logger)
    source_dataloader = source_provider.generate_dataloader('rag')
    src_prompt = generate_prompt(cfg, flag, dataset_src, source_dataloader, database, logger)
    src_json_path = save_dir + 'src.json'
    with open(src_json_path, 'w') as f:
        json.dump(src_prompt , f, indent=4)

    logger.info("generate src.json in {}".format(src_json_path))
    #####################################################################
    
    flag = 'target_train'
    database.update_mode(flag)
    target_provider = RoadDataProvider(cfg, flag=flag, logger=logger)
    target_dataloader = target_provider.generate_dataloader('rag')
    trg_prompt = generate_prompt(cfg, flag, dataset_src, target_dataloader, database, logger)
    trg_json_path = save_dir + 'trg.json'.format(dataset_src)
    with open(trg_json_path, 'w') as f:
        json.dump(trg_prompt , f, indent=4)

    logger.info("generate trg.json in {}".format(trg_json_path))
    #####################################################################

    flag = 'test'
    database.update_mode(flag)
    test_provider = RoadDataProvider(cfg, flag=flag, logger=logger)
    test_dataloader = test_provider.generate_dataloader('rag')
    test_prompt = generate_prompt(cfg, flag, dataset_src, test_dataloader, database, logger)
    test_json_path = save_dir + 'test.json'.format(dataset_src)
    with open(test_json_path, 'w') as f:
        json.dump(test_prompt , f, indent=4)
    
    logger.info("generate test.json in {}".format(test_json_path))

    
        

    


        

   