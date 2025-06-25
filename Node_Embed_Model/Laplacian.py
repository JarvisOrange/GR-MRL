import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import torch

# def _calculate_normalized_laplacian(adj):
#         adj = sp.coo_matrix(adj)
#         d = np.array(adj.sum(1))
#         d_inv_sqrt = np.power(d, -0.5).flatten()
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#         d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#         normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
#         return normalized_laplacian

def get_laplac_embed(adj_mx, dim=64):
        adj_matrix = adj_mx

        degrees = np.sum(adj_matrix, axis=1)
        D = np.diag(degrees)

        degrees_sqrt = np.sqrt(degrees)
        degrees_sqrt[degrees_sqrt == 0] = 1e-10  # 避免除零
        D_inv_sqrt = np.diag(1.0 / degrees_sqrt)
        L = np.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # 按特征值排序，选择前 dim 个非零特征值对应的特征向量
        idx = np.argsort(eigenvalues)[1:dim+1]  # 跳过第一个特征值（0）
        embedding = eigenvectors[:, idx]

        return embedding