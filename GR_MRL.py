import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from peft import get_peft_config, get_peft_model, TaskType, LoraConfig, AdaLoraConfig
from config import cfg
from Model.TSFormer.TSmodel import *
from Data.prompt_dataset import *


class GR_MRL(nn.Module):
    def __init__(self, mode='train'):
        super(GR_MRL, self).__init__()

        self.device = cfg['device']
        self.LLM_path = cfg['LLM_path']
        
        self.mode = 'train'

        temp, _= cfg['dataset_src_trg'].split('_')
        dataset_src = ''.join(temp.split('-'))

        self.load_time_encoder(dataset_src)

        self.build_VectorDatabase(dataset_src)

        self.set_LLM()

        self.set_

        self.


    def update_mode(self, mode='train'):
        self.mode = mode


    def load_time_encoder(self, dataset_src):
        model_path = './Save/pretrain_model/{}/best_model.pt'.format(dataset_src)
        self.time_encoder = TSFormer(cfg['TSFromer']).to(self.device)
        self.time_encoder.mode = 'test'


    def build_VectorDatabase(self, dataset_src):
        embed_path = './Save/time_embed/{}/embed.pt'.format(dataset_src)
        time_embed_pool = torch.load(embed_path).to(self.device)
        vd = VectorDataset(time_embed_pool)

    def set_LLM(self):
        llm_config = AutoConfig.from_pretrained(
            self.LLM_path / 'config.json',
            trust_remote_code = True,
            )
        
        self.word_hidden_size = llm_config['hidden_size']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LLM_path,
            trust_remote_code=True,
            )

        self.llm = AutoModel.from_pretrained(
             self.LLM_path,
            trust_remote_code=True
        ).cuda(self.device)


    def set_patch_embed_to_word_embed(self, time_embed):
        self.transform_layer = nn.Sequential(
            
        )


    def generate_prompt(self, batch):
        pass


    


    def forward(self, batch):
        with torch.zero_grad:
            time_embed = 





