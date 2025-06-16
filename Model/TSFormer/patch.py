import torch
import torch.nn as nn

class Patch(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.P = patch_size
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.input_embedding = nn.Conv2d(input_channel, output_channel, kernel_size=(self.P, 1), stride=(self.P, 1))

    def forward(self, input):

        B, C, L = input.shape
 
            
        input = input.unsqueeze(-1)             # B, C, L, 1
        
        # FC for (,p) to (,1)
        output = self.input_embedding(input)                         # B  d, L/P, 1
        output = output.squeeze(-1).view(B, self.output_channel, -1) # B  d, L/P,
        assert output.shape[-1] == L / self.P
        return output
