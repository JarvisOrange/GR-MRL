import torch
import torch.nn as nn

from config import cfg
from Model.TSFormer.TSmodel import *
from Data.VectorBase import *
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModel, AutoConfig, AutoTokenizer
from Model.MLoRA.peft import PeftModel, TaskType, get_peft_model
from Model.MLoRA.peft import MMOELoraSTConfig


class GR_MRL(nn.Module):
    def __init__(self, mode='source_train'):
        super(GR_MRL, self).__init__()

        self.device = cfg['device']
        self.LLM_path = cfg['LLM_path']
        
        self.mode = None

        temp, _= cfg['dataset_src_trg'].split('_')
        self.dataset_src = ''.join(temp.split('-'))


        self.time_pattern  = torch.load('Save/time_pattern/{}/embed_{}.pt'.format(self.dataset_src, cfg['time_cluster_k']))
        self.time_pattern.requires_grad = False

        self.road_pattern  = torch.load('Save/road_pattern/{}/embed_{}.pt'.format(self.dataset_src, cfg['time_cluster_k']))
        self.road_pattern.requires_grad = False

        # model part
        self.set_LLM()

        self.input_embed_dim = self.time_pattern.shape[1]
        self.mapping_layer = nn.Linear(self.time_pattern.shape[1], self.word_embed_dim)

        self.output_layer = nn.Linear(self.word_embed_dim, cfg['pre_num'])

        self.dropout = nn.Dropout(cfg['dropout'])

        self.update_embedding_layer()


    def update_embedding_layer(self):
        if self.mode == 'source_train':
            embed_path = './Save/time_embed/{}/embed_src.pt'.format(self.dataset_src)
        elif self.mode == 'target_train':
            embed_path = './Save/time_embed/{}/embed_trg.pt'.format(self.dataset_src)
        elif self.mode == 'test':
            embed_path = './Save/time_embed/{}/embed_test.pt'.format(self.dataset_src)
            

        time_embed_pool = torch.load(embed_path).to(self.device)
        num_embed, embed_dim = time_embed_pool.shape
        self.padding_idx = num_embed + 1

        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embed+1,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx
        )

        self.embedding_layer[:num_embed] = time_embed_pool

        self.embedding_layer.requires_grad = False
    

    def update_mode(self, mode='source_train'):
        self.mode = mode
        self.update_embedding_layer()


    def set_LLM(self):
        llm_config = AutoConfig.from_pretrained(
            self.LLM_path + '/config.json',
            trust_remote_code = True,
            )
        
        self.word_embed_dim = llm_config['hidden_size']
        
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

        lora_rank = cfg['lora_rank']
        lora_dropout = cfg['lora_dropout']
        lora_alpha = cfg['lora_alpha']

        if cfg['lora_method'] == 'moelora':
            TargetLoraConfig = MMOELoraSTConfig

            gate_embed_path = 'Save/time_pattern/{}/embed.pt'.format(self.dataset_src) + ';' \
                'Save/road_pattern/{}/embed.pt'.format(self.dataset_src)
                               
            kwargs = {
                  "gate_embed_dim": cfg['TSFormer']['out_dim'],
                  'gate_embed_path': gate_embed_path,
                  "expert_t_num": cfg['time_cluster_k'],
                  "expert_r_num": cfg['road_cluster_k'],
                  'top_k': cfg['top_k']
                  }
            
            task_type = TaskType.CAUSAL_LMS
            target_modules = cfg['target_modules'].split(',')
            modules_to_save = cfg['modules_to_save'].split(',')


        peft_config = TargetLoraConfig(
            task_type=task_type,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save,
            **kwargs
        )
        model = get_peft_model(model, peft_config)



    def attention_weighted_sum(self, vectors):
        #[b, d] * [d, k_t]
        time_scores = torch.matmul(vectors, self.time_pattern.transpose(0, 1))
        time_weights = F.softmax(time_scores, dim=1)  # [, k_t]
        
        #[b, d] * [d, k_r]
        road_scores = torch.matmul(vectors, self.road_pattern.transpose(0, 1))
        road_weights = F.softmax(road_scores, dim=1)  # [, k_r]
        
        time_weighted_sum = torch.matmul(time_weights, self.time_pattern)
        road_weighted_sum = torch.matmul(road_weights, self.road_pattern)

        return vectors + time_weighted_sum + road_weighted_sum


    def forward(self, batch_x):

        index,  prompts, ref = batch_x['index'], batch_x['ref'], batch_x['prompt']

        ref = [torch.tensor(r).to(self.device) for r in ref]

        ref = pad_sequence(ref, batch_first=True, padding_value=self.padding_idx) #b max_len 

        his_embed = self.embedding_layer(index) # b 1 d

        his_embed = self.attention_weighted_sum(his_embed)

        his_embed = self.mapping_layer(his_embed)

        ref_embed = self.embedding_layer(ref) # b * max_len * d

        ref_embed = self.mapping_layer(ref_embed)

        #dataset description and task description    
        prompt_desc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_desc_embed = self.llm.get_input_embeddings()(prompt_desc.to(self.device))   # (batch, prompt_token, dim)

        prompt_his = ['History information:'] * len(index)
        prompt_his = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_his_embed =  self.llm.get_input_embeddings()(prompt_his.to(self.device)) # (batch, prompt_token, dim)

        prompt_ref = ['Reference information:'] * len(index)
        prompt_ref = self.tokenizer(prompt_ref, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_ref_embed =  self.llm.get_input_embeddings()(prompt_ref.to(self.device)) # (batch, prompt_token, dim)

        llm_input = torch.hstack([prompt_desc_embed, prompt_his_embed, his_embed, prompt_ref_embed, ref_embed])

        llm_output = self.llm(inputs_embeds=llm_input).last_hidden_state

        output = self.output_layer(llm_output)

        return output



