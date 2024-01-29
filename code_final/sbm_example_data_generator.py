import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.sparse import save_npz
from scipy import stats
import os
import sys
import multilevel_final_new as mf
from bftry import BalancedForman
from lower_improve import LowerORicci

def lowercurv(G: nx.Graph):
    lrc = LowerORicci(G)
    G = lrc.lower_curvature()
    return G

def divide_group1(G: nx.Graph):
    # This function is only for 2 level hier
    blockn = nx.get_node_attributes(G, "block")
    for e in G.edges(data=True):
        if blockn[e[0]] == blockn[e[1]]:
            e[2]["group"] = 0 
        else:
            e[2]["group"] = 1 
    return G

def bfc2(G: nx.Graph, overbose="TRACE"):
    bfc = BalancedForman(G, verbose=overbose)
    G = bfc.compute_bfcsecond()
    return G


if __name__ == '__main__':
    
    num_nodes = 60 # total number of nodes
    num_communities = 2  # number of communities
    block_sizes = [num_nodes // num_communities] * num_communities  # equal sized communities
    p_matrix = [
        [0.8, 0.05],  
        [0.05, 0.8]  
    ]

    # Generate the SBM graph
    G = nx.stochastic_block_model(block_sizes, p_matrix)   
    G = mf.olliriccicurv(G)
    G = lowercurv(G)
    G = bfc2(G)
    G = divide_group1(G)
    edgelist = nx.to_pandas_edgelist(G)
    
    edgelist.to_csv('/data/edgelist.csv', index=False)