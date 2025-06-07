import os
import torch
from GR_MRL import GR_MRL
from config import cfg
from utils import kmeans_pytorch

def road_cluster(dataset_src, logger=None):
    device = cfg['device']

    