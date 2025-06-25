import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from tqdm import tqdm
import numpy as np 
import json
import torch
import numpy as np
import torch.nn.functional as F
import faiss
from torch.utils.data import Dataset, DataLoader



class PromptDataset(Dataset):
    def __init__(self, cfg, dataset_src, flag = 'src'):
        path = './Save/prompt/{}/{}.json'.format(dataset_src, flag)

        self.device = cfg['device']

        self.cfg = cfg
        
        with open(path, 'r') as f:
             self.prompt = json.load(f)
    

    def __len__(self):
        return len(self.prompt)
    

    def __getitem__(self, id):
        item = self.prompt[id]

        index =  torch.tensor(item['index']).to(self.device)
        ref = item['ref']
        prompt = item['prompt']
        label = item['label'].to(self.device)

        return {
            'index': index,
            'ref': ref,
            'prompt': prompt,
            'label': label
        }
        


class VectorBase():
    def __init__(self, dataset_src, vectors, mode='source_train'):
        self.mode= mode
        
        self.vectors = vectors

        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(self.vectors)

        self.load_related_json(dataset_src)

    def load_vector(self):
        pass

    def update_mode(self, mode):
        self.mode = mode
        


    def load_related_json(self, dataset_src):

        path = './Save/related/{}/result.json'.format(dataset_src)
        
        with open(path, 'r') as f:
             self.related_dict = json.load(f)

    
    def query(self, vector, k):
        q = vector.cpu().numpy()
        distances, indices = self.index.search(q, k+1)
        return [idx for idx in indices[0]]
    

    def query_related(self, q, k):
        temp_list =  self.query(q, k)

        res = []
        for i in temp_list:
            temp = self.related_dict.get(str(i), None)
            if temp != None:
                res += temp 
        return res
    

    def get_vector(self, index):
        return self.vectors[index].float().to(self.cfg['device'])
    

    
    