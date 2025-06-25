
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
from Exp.exp_pred import *

# B: PEMS_BAY
# L: MetrLA
# C Chengdu
# S: Shenzhen
# dataset name order: BLCS
# exmaple: B L C_S: B L C -> S

#Pretrain Data type
# dim=7 -> [speed, index_time_step, week_time, node, city, means, std]

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



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

    exp_dir = f"Checkpoints/exp_{cfg['exp_tag']}/{cfg['dataset_src_trg']}"

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    cfg['model_dir'] = exp_dir

    return cfg


def main():
    cfg = set_cfg()

    _logger = get_logger(cfg)

    temp, trg = cfg["dataset_src_trg"].split('_')
    dataset_src = ''.join(temp.split('-'))

    _logger.info('Start {}->{} Task'.format(dataset_src, trg))

    #stage 0:  preprocess data and get dataset 
    

    # #stage 1: pretrain time patch encoder or get pretrain time patch encoder
    exp_pretrain(cfg, logger=_logger)


    return
    
    # #stage 2: cluster time patch and node embedding
    exp_time_cluster(cfg, logger=_logger)
    exp_road_cluster(cfg, logger=_logger)


    # #stage 3: build time patch database
    exp_rag(cfg,logger=_logger)
    

    # #stage 4: finetune model
    exp_pred(cfg, logger=_logger)


if __name__ == '__main__':
    main()
