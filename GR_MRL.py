import torch
import torch.nn as nn

from Model.TSFormer.TSmodel import *
from Data.VectorBase import *
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModel, AutoConfig, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from Model.MLoRA.peft import PeftModel, TaskType, get_peft_model
from Model.MLoRA.peft import MMOELoraSTConfig


class GR_MRL(nn.Module):
    def __init__(self, cfg, mode='source_train'):
        super(GR_MRL, self).__init__()

        self.device = cfg['device']
        self.LLM_path = cfg['LLM_path']
        
        self.mode = mode
        
        self.max_length = cfg['GR_MRL']['max_length']

        temp, _= cfg['dataset_src_trg'].split('_')
        self.dataset_src = ''.join(temp.split('-'))


        self.time_pattern = torch.load('Save/time_pattern/{}/pattern_{}.pt'.format(self.dataset_src, cfg['time_cluster_k'])).cuda()
        self.time_pattern.requires_grad = False

        self.road_pattern = torch.load('Save/road_pattern/{}/pattern_{}.pt'.format(self.dataset_src, cfg['road_cluster_k'])).cuda()
        self.road_pattern.requires_grad = False

        self.retrieve_embed = torch.load('Save/time_pattern/{}/embed.pt'.format(self.dataset_src)).cuda()
        self.retrieve_embed.requires_grad = False

        # model part
        self.set_LLM(cfg)
        ###############

        self.retrieve_mapping_layer = nn.Linear(self.retrieve_embed.shape[1], self.word_embed_dim)

        self.retrieve_embedding_layer = nn.Embedding(
            num_embeddings=self.retrieve_embed.shape[0],
            embedding_dim=self.retrieve_embed.shape[1],
        )
        self.retrieve_embedding_layer.weight.data = self.retrieve_embed
        self.retrieve_embedding_layer.requires_grad = False
        self.retrieve_embedding_layer.to('cuda')

        self.mapping_layer = nn.Linear(self.time_pattern.shape[1], self.word_embed_dim)
        self.output_layer = nn.Linear(self.word_embed_dim, cfg['pre_num'])
        self.dropout = nn.Dropout(cfg['dropout'])

        self.update_embedding()


    def update_embedding(self):
        if self.mode == 'source_train':
            embed_path = './Save/time_embed/{}/embed_source.pt'.format(self.dataset_src)
        elif self.mode == 'target_train':
            embed_path = './Save/time_embed/{}/embed_target.pt'.format(self.dataset_src)
        elif self.mode == 'test':
            embed_path = './Save/time_embed/{}/embed_test.pt'.format(self.dataset_src)
            
        self.time_embed = torch.load(embed_path).to(self.device)
        
    def update_mode(self, mode='source_train'):
        self.mode = mode
        self.update_embedding()

    def set_LLM(self, cfg):
        llm_config = AutoConfig.from_pretrained(
            self.LLM_path,
            trust_remote_code = True,
            )
        
        self.word_embed_dim = llm_config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LLM_path,
            trust_remote_code=True,
            )

        special_token = {'additional_special_tokens': ['[PATCH]']}
        self.tokenizer.add_special_tokens(special_token)

        # Add quantization configuration (optional)
        use_quantization = cfg.get('use_quantization', False)
        
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.LLM_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                use_cache=False,
                trust_remote_code=True,
                quantization_config=quantization_config
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.LLM_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                use_cache=False,
                trust_remote_code=True
            )

        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.gradient_checkpointing_enable()


        lora_rank = cfg['lora_rank']
        lora_dropout = cfg['lora_dropout']
        lora_alpha = cfg['lora_alpha']

        if cfg['lora_method'] == 'stmoelora':
            gate_embed_path = 'Save/time_pattern/{}/pattern_{}.pt'.format(self.dataset_src, cfg['time_cluster_k']) + ';' \
                'Save/road_pattern/{}/pattern_{}.pt'.format(self.dataset_src, cfg['road_cluster_k'])
                               
            kwargs = {
                  "gate_embed_dim": cfg['TSFormer']['out_channel'],
                  'gate_embed_path': gate_embed_path,
                  "expert_t_num": cfg['time_cluster_k'],
                  "expert_r_num": cfg['road_cluster_k'],
                  'expert_top_k': cfg['expert_top_k']
                  }
            
            task_type = TaskType.CAUSAL_LMS
            target_modules = cfg['target_modules']
            modules_to_save = cfg['modules_to_save']

        peft_config = MMOELoraSTConfig(
            task_type=task_type,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save,
            **kwargs
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.llm.gradient_checkpointing_enable()

        self.peft_model = get_peft_model(self.llm, peft_config)

    def attention_weighted_sum(self, vectors):
        B, D = vectors.shape
        K_t, d = self.time_pattern.shape

        vectors = vectors.reshape(B, -1, d)
        time_scores = torch.matmul(vectors, self.time_pattern.transpose(0, 1))
        time_weights = F.softmax(time_scores, dim=-1)  # [B, 12, K_t]
        
        #[b, d] * [d, k_r]
        road_scores = torch.matmul(vectors, self.road_pattern.transpose(0, 1))
        road_weights = F.softmax(road_scores, dim=-1)  # [B, 12, K_t]
        
        time_weighted_sum = torch.matmul(time_weights, self.time_pattern)
        road_weighted_sum = torch.matmul(road_weights, self.road_pattern)

        temp_sum = vectors + time_weighted_sum + road_weighted_sum

        vectors_sum = temp_sum.reshape(B, -1)  # [B, D]

        return vectors_sum

    def forward(self, batch_x):
        
        #tobefix
        index = [item['index'] for item in batch_x]
        ref = [item['ref'] for item in batch_x]
        prompt = [item['prompt'] for item in batch_x]
        ref_length = [item['ref_length'] for item in batch_x]

        bs = len(index)

        tokenized = self.tokenizer(
            prompt,
            add_special_tokens=True,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=False
        )

        special_token = '[PATCH]'
        special_token_id = self.tokenizer.convert_tokens_to_ids(special_token)

        input_ids = tokenized['input_ids']

        patch_pos = []
        for i in range(bs):
            indices= []
            for index, value in enumerate(input_ids[i]):        
                if value == special_token_id:
                    indices.append(index)
            patch_pos.append(indices)

        input_ids = torch.tensor(input_ids).long().cuda()
        combined_embeds = self.llm.get_input_embeddings()(input_ids).cuda()

        with torch.amp.autocast(device_type='cuda'):
            for i in range(bs):
                for j in patch_pos[i]:
                    # j is [PATCH] position
                    assert input_ids[i][j] == special_token_id , ' mismatch'
                    patch_e = self.time_embed[j]
                    mapped_patch = self.mapping_layer(patch_e)
                    # Ensure mapped patch matches the embedding dtype
                    mapped_patch = mapped_patch.to(combined_embeds.dtype)
                    combined_embeds[i, j, :] = mapped_patch

            attention_mask = torch.tensor(tokenized['attention_mask']).cuda()
            
            # Ensure tensors match the embedding dtype for quantized models
            combined_embeds = combined_embeds.to(combined_embeds.dtype)
            attention_mask = attention_mask.to(combined_embeds.dtype)
            
            
            self.outputs = self.peft_model(
                input_ids = input_ids, #not use
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            last_hidden_state = self.outputs.hidden_states[-1]
            logits = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            output = self.output_layer(logits)
            # Ensure output is float32 for compatibility
            output = output.float()
            return output

        
    


