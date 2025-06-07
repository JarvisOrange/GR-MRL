from os import POSIX_FADV_DONTNEED
import torch
import torch.nn as nn

class LearnableTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)
        nn.init.uniform_(self.pe, -0.02, 0.02)
    
    def forward(self, X, index):
        """
        Args:
            x: Tensor, shape [B*N, L/P, d]
            index: Tensor, shape [B*N, L/P, 1]
        """
        if index is None:
            pe = self.pe[:X.size(1), :]
        else:
            pe = self.pe[index,:].squeeze(2)

        X  = X + pe
        X  = self.dropout(X)
        return X

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.tem_pe = LearnableTemporalPositionalEncoding(hidden_dim, dropout)

    def forward(self, input, index=None, abs_idx=None):
        """
        Args:
            input: B, N, L/P, d
            index: Tensor, shape [B, N, 1, L/P]
        """

        B, N, L_P, d = input.shape

        # temporal embedding
        index = index.contiguous().view(B*N,L_P,1)
        input = self.tem_pe(input.view(B*N, L_P, d), index=index)
        input = input.view(B, N, L_P, d)
        
        # absolute positional embedding
        return input


