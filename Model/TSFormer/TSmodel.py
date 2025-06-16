import torch
import torch.nn as nn
import gc
from TSFormer.Transformer_layers import TransformerLayers
from TSFormer.mask import MaskGenerator
from TSFormer.patch import Patch
from TSFormer.positional_encoding import PositionalEncoding


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class TSFormer(nn.Module):
    # def __init__(self, patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, L=6, mode='Pretrain', spectral=True):
    def __init__(self, model_cfg, mode='pretrain'):
        super().__init__()
        # patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, L, spectral = model_cfg['patch_size'], model_cfg['in_channel'], model_cfg['out_channel'], model_cfg['dropout'], model_cfg['mask_size'], model_cfg['mask_ratio'], model_cfg['L'], model_cfg['spectral']
        patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, n_layer = \
            model_cfg['patch_size'], model_cfg['in_channel'], model_cfg['out_channel'], \
            model_cfg['dropout'], model_cfg['mask_size'], model_cfg['mask_ratio'], model_cfg['n_layer']
        
        self.patch_size = patch_size
        self.seleted_feature = 0 # index of feature to reconstruct
        self.position_feature = 1 # index of time step
        self.mode = mode

        self.patch = Patch(patch_size, in_channel, out_channel, spectral=False)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        
        self.mask  = MaskGenerator(mask_size, mask_ratio)
        
        self.encoder = TransformerLayers(out_channel, n_layer)
        self.decoder = TransformerLayers(out_channel, 1)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        
        nn.init.uniform_(self.mask_token, -0.02, 0.02)
        
        self.output_layer = nn.Linear(out_channel, patch_size)

    def _forward_pretrain(self, input):
        # input : [B, 7, L]
        B, C, L = input.shape
        position = input[:,self.position_feature,:].unsqueeze(2)
        pos_indices = torch.arange(0, L, self.patch_size)
        position = position[:,:,pos_indices]

        # position : [B, 1, L/P]
        position = position // self.patch_size
    
        # B, 1, L
        input = input[:,self.seleted_feature,:].unsqueeze(2)

        # get patches and exec input embedding
        patches = self.patch(input)             # B, d, L/P
        patches = patches.transpose(-1, -2)     # B, L/P, d
        
        # positional embedding
        # patches : [B, L/P, d]
        # position : [B, 1, L/P]
        patches = self.pe(patches, index=position.long()) # (B, L_P, d)
        
        
        
        # mask tokens
        # both 1D vector contains the index
        # 25, 75
        unmasked_token_index, masked_token_index = self.mask()

        encoder_input = patches[:, :, unmasked_token_index, :]        

        # encoder
        H = self.encoder(encoder_input)         # B, L/P*(1-r), d

        # encoder to decoder
        H = self.encoder_2_decoder(H)           # B, L/P*(1-r), d
        
        H_unmasked = H


        # arg1 : [B,  len(mti), d].  arg2 : [B, 1, L/P]
        masked_token_index_inpe = torch.tensor(masked_token_index)
        # position : [B, 1, len(mask_token_index)]
        indices = masked_token_index_inpe.expand(B,  1, len(masked_token_index))
        # pe input : patches : [B, len(mti), d]. position : [B, 1, len(mti)]
        H_masked = self.pe(self.mask_token.expand(B,  len(masked_token_index_inpe), H.shape[-1]), index=indices.long())
        
        
        # B,  L/P, d
        H_full = torch.cat([H_unmasked, H_masked], dim=-2)   
        # B, L/P, d

        # decoder
        H = self.decoder(H_full)

        
        # output layer
        # B,  L/P, P
        out_full = self.output_layer(H)

        # prepare loss
        B,  _, _ = out_full.shape 
        # B, len(mask), P
        out_masked_tokens = out_full[:,  len(unmasked_token_index):, :]
        # B, len(mask) * P, N
        out_masked_tokens = out_masked_tokens.view(B, -1)
        
    
        # B, 1, L -> B, L, 1 -> B, L/P, 1, P -> B, L/P, P 
        label_full  = input.permute(0, 2, 1).unfold(1, self.patch_size, self.patch_size).squeeze(2)  # B, L/P, P
        # B, L/P * r, P
        label_masked_tokens  = label_full[:,  masked_token_index, :].contiguous()
        # B, N, L/p * r * P -> B, L/p * r * P
        label_masked_tokens  = label_masked_tokens.view(B, -1)
        
        
        # return
        # torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r]
        # torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r]
        return out_masked_tokens, label_masked_tokens

    def _inference(self, input):
        """the feed forward process in the forecasting stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.

        Returns:
            torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """

        B, C, L = input.shape
        position = input[:,:,self.position_feature,:].unsqueeze(2)
        pos_indices = torch.arange(0, L, self.patch_size)
        position = position[:,:,:,pos_indices]

        # position : [B,  1, L/P]
        position = position // 12
        position = position % 168 # 168 = 24 * 7

        # B, 1, L
        input = input[:,:,self.seleted_feature,:].unsqueeze(2)

        # get patches and exec input embedding
        patches = self.patch(input)             # B, d, L/P
        patches = patches.transpose(-1, -2)     # B,  L/P, d
        
        # positional embedding
        # patches : [B, N, L/P, d]. position : [B,  1, L/P]
        patches = self.pe(patches,position.long())
        
        encoder_input = patches

        # encoder
        H = self.encoder(encoder_input)         # B,  L/P, d

        H = H.reshape(B, -1)
        return H
    

    def forward(self, input_data):
        """feed forward of the TSFormer.
        TSFormer has two modes: the pre-training mode and the test mode, which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            input_data (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.
        
        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
                torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
                dict: data for plotting.
            forecasting: 
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """
        if self.mode == 'pretrain':
            return self._forward_pretrain(input_data)
        elif self.mode == 'test':
            return self._inference(input_data)
