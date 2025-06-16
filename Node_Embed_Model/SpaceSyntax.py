import momepy
import networkx as nx
import numpy as np


def min_max_scale(arr):
    """手动实现 0-1 标准化"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:  # 防止除以零
        return arr - min_val
    return (arr - min_val) / (max_val - min_val)

def get_connectivity(g):
    temp_g = momepy.node_degree(g)
    temp_g = temp_g.nodes(data=True) #[(0, {'degree': 3}), (2, {'degree': 7}), (3, {'degree': 3}), (1, {'degree': 4})]
    temp_g = sorted(temp_g,key=lambda x: x[0])

    l = []
    for item in temp_g:
        l.append(item[1]['degree'])

    l = np.array(l)
    return min_max_scale(l)

def get_integration(g):
    temp_g = momepy.closeness_centrality(g)
    temp_g =  temp_g.nodes(data=True)
    temp_g = sorted(temp_g,key=lambda x: x[0])

    l = []
    for item in temp_g:
        l.append(item[1]['closeness'])

    l = np.array(l)
    return min_max_scale(l)


def get_choice(g):
    temp_g =  momepy.betweenness_centrality(g)
    temp_g =  temp_g.nodes(data=True)
    temp_g = sorted(temp_g,key=lambda x: x[0])

    l = []
    for item in temp_g:
        l.append(item[1]['betweenness'])

    l = np.array(l)
    return min_max_scale(l)

def make_directed_graph(adj_mx):
    """
    将邻接矩阵转换为有向图
    :param adj_mx: 邻接矩阵
    :return: 有向图
    """
    temp_list = list(adj_mx)
    DiG = nx.MultiDiGraph()
    for i in range(len(temp_list)):
        for j in range(len(temp_list[i])):
            if adj_mx[i][j] != 0:
                DiG.add_edge(i, j, key='A', mm_len=adj_mx[i][j])
    
    return DiG

def get_space_syntax_embed(adj_mx):
    DiG = make_directed_graph(adj_mx)

    connectivity = get_connectivity(DiG)
    integration = get_integration(DiG)
    choice = get_choice(DiG)
    
    space_syntax_embed = np.vstack((connectivity, integration, choice))
    
    return space_syntax_embed.T #