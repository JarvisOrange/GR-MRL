from tqdm import tqdm
import numpy as np 
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from config import cfg


class VectorDatabase():
    def __init__(self, vectors):
        self.vectors = vectors.numpy()

    
    #求向量的余弦相似度，传入两个向量和一个embedding模型，返回一个相似度
    def get_similarity(self, vector1, vector2):
        return F.cossine_similarity(vector1, vector2, dim=0)
    
    #求一个字符串和向量列表里的所有向量的相似度，表进行排序，返回相似度前k个的子块列表
    def query(self, query,  k):
        result = np.array([self.get_similarity(query, vector) for vector in self.vectors])
        return np.array(self.vectors)[result.argsort()[-k:][::-1]].tolist()
    
