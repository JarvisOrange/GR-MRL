
import logging
import os
from argparse import ArgumentParser
import json

from GR_MRL import GR_MRL
from utils import *
from Exp.exp_pretrain import *
from Exp.exp_rag import *
from Exp.exp_road_cluster import *
from Exp.exp_time_cluster import *
from Exp.exp_ft import *

# B: PEMS_BAY
# L: MetrLA
# C Chengdu
# S: Shenzhen
# dataset name order: BLCS
# exmaple: B L C_S: B L C -> S

#Pretrain Data type
# dim=7 -> [speed, index_time_step, week_time, node, city, means, std]

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

stage_dict = {
    0: 'all_process',
    1: 'pretrain',
    2: 'cluster',
    3: 'rag',
    4: 'finetune'
}



def set_cfg():
    parser = ArgumentParser()
    parser.add_argument("--exp_tag", type=str)
    parser.add_argument("--dataset_src_trg", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()


    with open('./config.json', 'r', encoding='utf-8') as file:
        cfg = json.load(file)
        
    
    # cfg["exp_tag"] = args.exp_tag
    # cfg["dataset_src_trg"] = args.dataset_src_trg
    # cfg["device"] = args.device

    exp_dir = f"Checkpoints/exp__{cfg['exp_tag']}/{cfg['dataset_src_trg']}"

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    cfg['model_dir'] = exp_dir

    return cfg


def main():
    cfg = set_cfg()

    _logger = get_logger(cfg)

    temp, trg = cfg["dataset_src_trg"].split('_')
    dataset_src = ''.join(temp.split('-'))

    _logger.info('Task: {}--->{} '.format(dataset_src, trg))

    stage = cfg['stage']


    # stage 0: all process
    if stage == 0:
        _logger.info('<<<<<---------- pretrain ---------->>>>>'.format(stage_dict[stage]))
        exp_pretrain(cfg, logger=_logger)

        _logger.info('<<<<<---------- cluster ---------->>>>>'.format(stage_dict[stage]))
        exp_time_cluster(cfg, logger=_logger)
        exp_road_cluster(cfg, logger=_logger)

        _logger.info('<<<<<---------- rag ---------->>>>>'.format(stage_dict[stage]))
        exp_rag(cfg,logger=_logger)

        _logger.info('<<<<<---------- finetune ---------->>>>>'.format(stage_dict[stage]))
        exp_ft(cfg, logger=_logger)
    

    ########################################################################

    # stage 1: pretrain time patch encoder or get pretrain time patch encoder
    if stage == 1:
        _logger.info('<<<<<---------- {} Start---------->>>>>'.format(stage_dict[stage]))
        exp_pretrain(cfg, logger=_logger)
        _logger.info('<<<<<---------- {} End---------->>>>>'.format(stage_dict[stage]))

    # stage 2: cluster time patch and node embedding
    elif stage == 2:
        _logger.info('<<<<<---------- {} Start---------->>>>>'.format(stage_dict[stage]))
        exp_time_cluster(cfg, logger=_logger)
        exp_road_cluster(cfg, logger=_logger)
        _logger.info('<<<<<---------- {} End---------->>>>>'.format(stage_dict[stage]))

    # stage 3: build time patch database
    elif stage == 3:
        _logger.info('<<<<<---------- {} Start---------->>>>>'.format(stage_dict[stage]))
        exp_rag(cfg,logger=_logger)
        _logger.info('<<<<<---------- {} End---------->>>>>'.format(stage_dict[stage]))
    
    # stage 4: finetune model
    elif stage == 4:
        _logger.info('<<<<<---------- {} Start---------->>>>>'.format(stage_dict[stage]))
        exp_ft(cfg, logger=_logger)
        _logger.info('<<<<<---------- {} End---------->>>>>'.format(stage_dict[stage]))


if __name__ == '__main__':
    main()
