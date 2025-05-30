import momepy
import osmnx as ox
import networkx as nx
import torch
import numpy as np


def get_connectivity(g):
    temp_g = momepy.node_degree(g)
    print(temp_g.edges(data=True))

def get_integration(g):
    temp_g = momepy.closeness_centrality(g)
    print(temp_g.edges(data=True))

def get_choice(g):
    temp_g =  momepy.betweenness_centrality(g)
    print(temp_g.edges(data=True))



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
                DiG.add_edge(i, j,  key='A', mm_len=adj_mx[i][j])

    return DiG

def get_space_syntax_embed(adj_mx):
    DiG = make_directed_graph(adj_mx)
   

    connectivity = get_connectivity(DiG)
    integration = get_integration(DiG)
    choice = get_choice(DiG)
    
    
  
    

l = ['CD','BAY','LA','SZ']
matrix = np.load('raw_data/'+ l[3] +'/matrix.npy')
print(matrix.shape)
get_space_syntax_embed(matrix)