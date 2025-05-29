
import logging
import os
from argparse import ArgumentParser

from config import cfg
from GR_MRL import GR_MRL

def set_log(exp_tag):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s](%(asctime)s):%(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler = logging.FileHandler('log/' + "exp_" + str(exp_tag) +  '.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s](%(asctime)s):%(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


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

    exp_dir = f"checkpoint/exp_{cfg['exp_tag']}/{args.src}_to_{args.trg}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    cfg['model_dir'] = exp_dir


def main():
    set_log(cfg['exp_tag'])
    def some_function():
        global logger  # 可选：若需要显式声明
        logger.info("This is a log message")
    set_cfg()
    model = GR_MRL(cfg)
    model.to(cfg['device'])

    #stage 0
    dataset_preprocess()
    
    #stage 1
    pretrain_time_encoder()
    
    #stage 2
    cluster_node_embed()
    
    #stage 3
    exp_finetune()

    # Cost Prediction
    disentangle_train(model, save_model=True)

    # Preference Learning
    network = get_network(cfg['dataset_source'])
    train_traj, valid_traj = get_traj_dataset(cfg['dataset_source'])
    preference_train(model, network, train_traj, valid_traj, save_model=True)


if __name__ == '__main__':
    main()
