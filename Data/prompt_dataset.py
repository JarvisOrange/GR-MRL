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

class VectorDataset():
    def __init__(self, vectors, cfg):
        self.vectors = vectors
        self.vectors.requires_grad = False

        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(self.vectors)

    def load_json(self,cfg):
        temp, _= cfg['dataset_src_trg'].split('_')
        dataset_src = ''.join(temp.split('-'))

        path = './Save/road_related/{}/result.json'.format(dataset_src)
        
        with open(path, 'r') as f:
             self.related_dict = json.load(f)

    
    def query(self, q,  k):
        distances, indices = self.index.search(q, k)
        return [idx for idx in indices[0]]
    

    def query_related(self, q, k):
        temp_list =  self.query(q, k)

        res = []
        for i in temp_list:
            res += self.related_dict[i]
        return res
    

    def get_vector(self, index):
        return self.vectors[index].float().to(cfg['device'])
    

    def generate_prompt_dataset(self, encode_model, dataloader):

        prompt_dict = {}

        k = cfg['retrieve_k']

        for index in range(self.vectors.shape[0]):

            related_list = self.query_related(self.vectors[index], k)

            prompt = 
    