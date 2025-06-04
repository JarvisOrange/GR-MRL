
import logging
import os
from argparse import ArgumentParser

from config import cfg
from GR_MRL import GR_MRL
from utils import *



def set_cfg():
    parser = ArgumentParser()
    parser.add_argument("--exp_tag", type=int)
    parser.add_argument("--src", type=str)
    parser.add_argument("--trg", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    
    cfg["exp_tag"] = args.exp_tag
    cfg["dataset_source"] = args.src
    cfg["dataset_target"] = args.trg
    cfg["device"] = args.device

    exp_dir = f"checkpoint/exp_{cfg['exp_tag']}/{args.src}_{args.trg}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    cfg['model_dir'] = exp_dir


def main():
    set_cfg()
    _logger = get_logger(cfg)
    
    model = GR_MRL(cfg)
    model.to(cfg['device'])

    #stage 0:  preprocess data and get dataset 
    source_dataset = RoadTimeDataset(cfg, _logger)
    finetune_dataset = RoadTimeDataset(cfg, _logger)
    test_dataset = RoadTimeDataset(cfg, _logger)


    
    # #stage 1: pretrain time patch encoder or get pretrain time patch encoder
    # time_patch_encoder = TimePatchEncoder()
    # time_patch_encoder.train()
    
    # #stage 2: cluster time patch and node embedding
    # cluster_time_embed()
    # cluster_node_embed()
    
    # #stage 3: build time patch database
    # make_time_patch_database()

    # #stage 4: finetune model
    # generate_prompt()

    # finetune_model()


if __name__ == '__main__':
    main()
