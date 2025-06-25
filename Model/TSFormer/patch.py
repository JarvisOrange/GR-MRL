import torch
import torch.nn as nn

class Patch(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.P = patch_size
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.input_embedding = nn.Conv1d(input_channel, output_channel, kernel_size=self.P, stride=self.P)

    def forward(self, input):
        # B, d, L
        B, C, L = input.shape
 
        output = self.input_embedding(input)                         # B  d, L/P
        assert output.shape[-1] == L / self.P
        return output
