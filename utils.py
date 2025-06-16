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
import torch.nn as nn



def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


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
    log_filename = '{}-{}-{}.log'.format(config['exp_tag'],
                                             config['dataset_src_trg'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    


def kmeans_pytorch(X, n_clusters, max_iter=100, tol=1e-4, device=None):
    """
    PyTorch 实现的 K-means 聚类（GPU 版本）
    """
    X = X.to(device)

    n_samples, n_features = X.shape
    indices = torch.randperm(n_samples, device=device)[:n_clusters]
    centers = X[indices].clone()

    for _ in range(max_iter):
        distances = torch.cdist(X, centers)  # [n_samples, n_clusters]
        
        labels = torch.argmin(distances, dim=1)  # [n_samples]
        
        new_centers = torch.zeros_like(centers)
        for i in range(n_clusters):
            mask = (labels == i)
            if torch.sum(mask) > 0:
                new_centers[i] = torch.mean(X[mask], dim=0)
        
        
        if torch.norm(new_centers - centers) < tol:
            break
            
        centers = new_centers

    return labels.cpu(), centers.cpu()


def calc_metric(pred, y, flag = "train"):
    # input B * L * N

    B, L, N = pred.shape
    pred = pred.transpose(B, N, -1).reshape(-1,L) # (B * N, L)
    y = y.transpose(B,N,-1).reshape(-1,L)
    MSE = torch.mean((pred - y)**2, dim = 0)
    RMSE = torch.sqrt(MSE)
    MAE = torch.mean(torch.abs(pred - y), dim = 0)
    MAPE = torch.mean(torch.abs(pred - y) / y, dim = 0)
    return MSE, RMSE, MAE, MAPE


def unnorm(x ,means, stds):
    B, LL, N = x.shape
    means = means.expand(B, LL, N)
    stds = stds.expand(B, LL, N)
    return x * stds + means

def adj_to_dict(adj):
    #!!! CD SZ may have weight which is not 1 or 0
    d = {}
    l1, l2 = adj.shape
    adj = adj.to_list()

    for i in range(l1):
        for j in range(l2):
            if adj[i][j] != 0:
                if d[i] == None: 
                    d[i] = [j]
                else: 
                    d[i] += d[i] + [j]
    return d
                
    

class Feq_Loss(nn.Module):
    def __init__(self):
            super(Feq_Loss, self).__init__()
            
    def forward(self, y_pred, y_true):
        loss_feq = (torch.fft.rfft(y_pred, dim=1) - torch.fft.rfft(y_true, dim=1)).abs().mean()
        return loss_feq
    



