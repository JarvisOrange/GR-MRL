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

        self.set_LLM()

        self.time_pattern  = torch.load('Save/time_pattern/{}/embed_{}.pt'.format(dataset_src, cfg['time_cluster_k']))
        self.time_pattern.requires_grad = False
        
        self.road_pattern  = torch.load('Save/road_pattern/{}/embed_{}.pt'.format(dataset_src, cfg['time_cluster_k']))
        self.road_pattern.requires_grad = False

        

        self.mapping_layer = nn.Linear(self.)


    def update_mode(self, mode='train'):
        self.mode = mode


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

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token


    def generate_prompt(self, batch):
        pass


    


    def forward(self, batch):


        # with torch.zero_grad:
        #     time_embed = 

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state



