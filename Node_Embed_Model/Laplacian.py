import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


def get_laplacian_node_representations(graph, num_eigenvectors=10):
    """
    使用拉普拉斯方法获取图节点表征
    :param graph: networkx 图对象
    :param num_eigenvectors: 要保留的特征向量数量
    :return: 节点表征矩阵
    """
    # 计算图的拉普拉斯矩阵
    laplacian_matrix = nx.laplacian_matrix(graph).asfptype()

    # 计算拉普拉斯矩阵的特征值和特征向量
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_eigenvectors, which='SM')

    # 特征向量即为节点表征
    node_representations = eigenvectors

    return node_representations


# 示例使用
if __name__ == "__main__":
    # 创建一个简单的图
    G = nx.karate_club_graph()

    # 获取节点表征
    node_representations = get_laplacian_node_representations(G, num_eigenvectors=10)

    print("节点表征矩阵的形状:", node_representations.shape)
    print("前几个节点的表征:")
    print(node_representations[:5])
    