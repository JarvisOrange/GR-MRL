import numpy as np
import torch

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_norm = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_norm
def is_symmetric(matrix):
    """
    检查矩阵是否为对称矩阵
    
    参数:
    matrix: 二维列表或NumPy数组
    返回:
    bool: 是否为对称矩阵
    """
    # 转换为NumPy数组（处理列表输入）
    mat = np.array(matrix)
    
    # 检查是否为方阵
    if mat.shape[0] != mat.shape[1]:
        return False
    
    # 比较矩阵与转置（考虑浮点数精度）
    if np.allclose(mat, mat.T):
        return True
    else:
        return False
    
l = ['CD','BAY','LA','SZ']
matrix = np.load('raw_data/'+ l[3] +'/matrix.npy')
A = torch.from_numpy(matrix).float()
print(A.shape)
print(is_symmetric(A))