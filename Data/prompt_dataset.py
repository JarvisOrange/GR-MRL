from tqdm import tqdm
import numpy as np 
import json
import torch
import numpy as np
import torch.nn.functional as F
from config import cfg
import faiss
from torch.utils.data import Dataset, DataLoader

class PromptDataset():
    def __init__(self):
        pass
    

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class VectorBase():
    def __init__(self, dataset_src, vectors):
        self.vectors = vectors
        self.vectors.requires_grad = False

        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(self.vectors)

        self.load_related_json(dataset_src)

        self.flag = 'source_train'


    def load_related_json(self, dataset_src):

        path = './Save/road_related/{}/result.json'.format(dataset_src)
        
        with open(path, 'r') as f:
             self.related_dict = json.load(f)

    
    def query(self, q,  k, index):
        
        if self.flag != 'source_train':
            distances, indices = self.index.search(q, k)
            return [idx for idx in indices[0]]
        else:
            distances, indices = self.index.search(q, k+1)
            return [idx for idx in indices[0] if idx != index]
    

    def query_related(self, index, q, k):
        temp_list =  self.query(q, k, index)

        res = []
        for i in temp_list:
            res += self.related_dict[i]
        return res
    

    def get_vector(self, index):
        return self.vectors[index].float().to(cfg['device'])
    

    
    