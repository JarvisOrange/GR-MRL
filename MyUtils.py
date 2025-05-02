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

class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行 > config file > default config
    """

    def __init__(self, dataset, config_file=None,
                 saved_model=True, train=True, other_args=None):
        """
        Args:
            task, model, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        self.config = {}
        self._parse_external_config(dataset, saved_model, train, other_args)
        self._parse_config_file(config_file)
        self._init_device()

    def _parse_external_config(self, dataset,
                               saved_model=True, train=True, other_args=None):
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: 对 config file 的格式进行检查
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))
            
        # 加载数据集config.json
        with open('./raw_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu:
            torch.cuda.set_device(gpu_id)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config





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


class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



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


class BatchPAD(Batch):

    def __init__(self, feature_name, pad_item=None, pad_max_len=None):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
            pad_item (dict): key is the feature name, and value is the padding
            value. We will just padding the feature in pad_item
            pad_max_len (dict): key is the feature name, and value is the max
                length of padded feature. use this parameter to truncate the
                feature.
        """
        super().__init__(feature_name=feature_name)
        # 默认是根据 batch 中每个特征最长的长度来补齐，如果某个特征的长度超过了 pad_max_len 则进行剪切
        self.pad_len = {}
        self.origin_len = {}  # 用于得知补齐前轨迹的原始长度
        self.pad_max_len = pad_max_len if pad_max_len is not None else {}
        self.pad_item = pad_item if pad_item is not None else {}
        for key in feature_name:
            self.data[key] = []
            if key in self.pad_item:
                self.pad_len[key] = 0
                self.origin_len[key] = []

    def append(self, item):
        """
        append a new item into the batch

        Args:
            item (list): 一组输入，跟feature_name的顺序一致，feature_name即是这一组输入的名字
        """
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            # 需保证 item 每个特征的顺序与初始化时传入的 feature_name 中特征的顺序一致
            self.data[key].append(item[i])
            if key in self.pad_item:
                self.origin_len[key].append(len(item[i]))
                if self.pad_len[key] < len(item[i]):
                    # 保持 pad_len 是最大的
                    self.pad_len[key] = len(item[i])

    def padding(self):
        """
        只提供对一维数组的特征进行补齐
        """
        for key in self.pad_item:
            # 只对在 pad_item 中的特征进行补齐
            if key not in self.data:
                raise KeyError('when pad a batch, raise this error!')
            max_len = self.pad_len[key]
            if key in self.pad_max_len:
                max_len = min(self.pad_max_len[key], max_len)
            for i in range(len(self.data[key])):
                if len(self.data[key][i]) < max_len:
                    self.data[key][i] += [self.pad_item[key]] * \
                        (max_len - len(self.data[key][i]))
                else:
                    # 截取的原则是，抛弃前面的点
                    # 因为是时间序列嘛
                    self.data[key][i] = self.data[key][i][-max_len:]
                    # 对于剪切了的，我们没办法还原，但至少不要使他出错
                    self.origin_len[key][i] = max_len

    def get_origin_len(self, key):
        return self.origin_len[key]

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
            elif self.feature_name[key] == 'array of int':
                for i in range(len(self.data[key])):
                    for j in range(len(self.data[key][i])):
                        try:
                            self.data[key][i][j] = torch.LongTensor(np.array(self.data[key][i][j])).to(device)
                        except TypeError:
                            print('device is ', device)
                            exit()
            elif self.feature_name[key] == 'no_pad_int':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.LongTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_pad_float':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.FloatTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_tensor':
                pass
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)