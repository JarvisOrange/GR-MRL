import networkx as nx
import numpy as np
import random
import os
from tqdm import tqdm
import time
import sys
import json
import torch
import logging
import datetime
from torch.utils.data import Dataset
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    return None

def get_dataset_list(data_list):
    dlist = []
    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(config, name=None):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}.log'.format(config['dataset'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = config.get('log_level', 'INFO')

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def get_device(args):
    gpu = args.gpu
    return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')


def get_optimizer(model, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)




class Batch(object):

    def __init__(self, feature_name):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
        """
        self.data = {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        """
        append a new item into the batch

        Args:
            item (list): 一组输入，跟feature_name的顺序一致，feature_name即是这一组输入的名字
        """
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            self.data[key].append(item[i])

    def to_tensor(self, device):
        """
        将数据self.data转移到device上

        Args:
            device(torch.device): GPU/CPU设备
        """
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))

    def to_ndarray(self):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = np.array(self.data[key])
            elif self.feature_name[key] == 'float':
                self.data[key] = np.array(self.data[key])
            else:
                raise TypeError(
                    'Batch to_ndarray, only support int, float but you give {}'.format(self.feature_name[key]))
 

class Feq_Loss(nn.Module):
    def __init__(self):
            super(Feq_Loss, self).__init__()
            
    def forward(self, y_pred, y_true):
        loss_feq = (torch.fft.rfft(y_pred, dim=1) - torch.fft.rfft(y_true, dim=1)).abs().mean()
        return loss_feq
    