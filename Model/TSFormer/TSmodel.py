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
        """feed forward of the TSFormer in the pre-training stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 2, L * P.
                                The first dimension is speed. The second dimension is position.
nn
        Returns:
            torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
            torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
            dict: data for plotting.
        """
        # input : [B, N, 7, L]
        B, N, C, L = input.shape
        position = input[:,:,self.position_feature,:].unsqueeze(2)
        pos_indices = torch.arange(0, L, self.patch_size)
        position = position[:,:,:,pos_indices]

        # position : [B, N, 1, L/P]
        position = position // self.patch_size
    
        # B, N, 1, L
        input = input[:,:,self.seleted_feature,:].unsqueeze(2)

        # get patches and exec input embedding
        patches = self.patch(input)             # B, N, d, L/P
        patches = patches.transpose(-1, -2)     # B, N, L/P, d
        
        # positional embedding
        # patches : [B, N, L/P, d]
        # position : [B, N, 1, L/P]
        patches = self.pe(patches, index=position.long()) # (B, N, L_P, d)
        
        
        
        # mask tokens
        # both 1D vector contains the index
        # 25, 75
        unmasked_token_index, masked_token_index = self.mask()

        encoder_input = patches[:, :, unmasked_token_index, :]        

        # encoder
        H = self.encoder(encoder_input)         # B, N, L/P*(1-r), d

        # encoder to decoder
        H = self.encoder_2_decoder(H)           # B, N, L/P*(1-r), d
        
        H_unmasked = H


        # arg1 : [B, N, len(mti), d].  arg2 : [B, N, 1, L/P]
        masked_token_index_inpe = torch.tensor(masked_token_index)
        # position : [B, N, 1, len(mask_token_index)]
        indices = masked_token_index_inpe.expand(B, N, 1, len(masked_token_index))
        # pe input : patches : [B, N, len(mti), d]. position : [B, N, 1, len(mti)]
        H_masked = self.pe(self.mask_token.expand(B, N, len(masked_token_index_inpe), H.shape[-1]), index=indices.long())
        
        
        # B, N, L/P, d
        H_full = torch.cat([H_unmasked, H_masked], dim=-2)   
        # B, N, L/P, d

        # decoder
        H = self.decoder(H_full)

        
        # output layer
        # B, N, L/P, P
        out_full = self.output_layer(H)

        # prepare loss
        B, N, _, _ = out_full.shape 
        # B, N, len(mask), P
        out_masked_tokens = out_full[:, :, len(unmasked_token_index):, :]
        # B, len(mask) * P, N
        out_masked_tokens = out_masked_tokens.view(B, N, -1).transpose(1, 2)
        
    
        # B, N, 1, L -> B, L, N, 1 -> B, L/P, N, 1, P -> B, L/P, N, P -> B, N, L/P, P
        label_full  = input.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.seleted_feature, :].transpose(1, 2)  # B, N, L/P, P
        # B, N, L/P * r, P
        label_masked_tokens  = label_full[:, :, masked_token_index, :].contiguous()
        # B, N, L/p * r * P -> B, L/p * r * P, N
        label_masked_tokens  = label_masked_tokens.view(B, N, -1).transpose(1, 2)
        
        
        # return
        # torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
        # torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
        return out_masked_tokens, label_masked_tokens

    def _inference(self, input):
        """the feed forward process in the forecasting stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.

        Returns:
            torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """

        B, N, C, L = input.shape
        position = input[:,:,self.position_feature,:].unsqueeze(2)
        pos_indices = torch.arange(0, L, self.patch_size)
        position = position[:,:,:,pos_indices]

        # position : [B, N, 1, L/P]
        position = position // 12
        position = position % 168 # 168 = 24 * 7

        # B, N, 1, L
        input = input[:,:,self.seleted_feature,:].unsqueeze(2)

        # get patches and exec input embedding
        patches = self.patch(input)             # B, N, d, L/P
        patches = patches.transpose(-1, -2)     # B, N, L/P, d
        
        # positional embedding
        # patches : [B, N, L/P, d]. position : [B, N, 1, L/P]
        patches = self.pe(patches,position.long())
        
        encoder_input = patches

        # encoder
        H = self.encoder(encoder_input)         # B, N, L/P, d
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
        if self.mode == 'pain':
            return self._forward_pretrain(input_data)
        elif self.mode == 'test':
            return self._inference(input_data)
