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
import orjson



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
    def __init__(self, cfg, dataset_src, vectors, mode='source_train'):
        self.mode= mode
        
        self.vectors = vectors
        
        device_id = int(cfg['device'].split(':')[1])

        index_cpu = faiss.IndexFlatL2(self.vectors.shape[1])

        index_gpu = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),  
            device_id,
            index_cpu
            )
        index_gpu.add(self.vectors.cpu())

        self.v_index = index_gpu

        self.load_related_json(dataset_src)

    def load_vector(self):
        pass

    def update_mode(self, mode):
        self.mode = mode
        

    def load_related_json(self, dataset_src):

        path = './Save/related/{}/result.json'.format(dataset_src)
        
        self.related_dict = {}
        with open(path, "r") as f:
            self.related_dict = orjson.loads(f.read())

    
    def query(self, vector, k):
        q = vector.cpu()
        distances, indices = self.v_index.search(q, k)
        return [idx[0] for idx in indices]
    

    def query_related(self, q, k):
        
        temp_list =  self.query(q, k)
        
        res = {}
        for i in range(q.shape[0]):
            temp = self.related_dict.get(str(temp_list[i]), None)
            if temp != None:
                res[i] = temp
        return res
    

    def get_vector(self, index):
        return self.vectors[index].float().to(self.cfg['device'])
    

    
    