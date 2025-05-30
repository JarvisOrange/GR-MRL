import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import torch

def _calculate_normalized_laplacian(adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian

def get_laplac_embed(adj_mx, dim=128):
        L, isolated_point_num = _calculate_normalized_laplacian(adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe

