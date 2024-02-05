import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.sparse import save_npz
from scipy import stats
import os
import sys
sys.path.insert(1, "curvatures")
from balanced_forman_curvature import BalancedForman
from lower_ricci_curvature import LowerORicci


def olliriccicurv(G: nx.Graph, oalpha = 0.5, overbose="TRACE"):
    """Compute Ollivier Ricci Curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.
    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with "ricciCurvature" on nodes and edges.
    """
    orc = OllivierRicci(G, alpha=oalpha, verbose=overbose)
    G = orc.compute_ricci_curvature()
    return G

def lowercurv(G: nx.Graph):
    """
    A function to calculate and save LRC value for every edge in a graph.
    
    Parameters
    ----------
    G: A Networkx graph.
    
    Returns
    -------
    G: A Networkx graph with lrc value saved in each edge attribute. 
    """
    lrc = LowerORicci(G)
    G = lrc.compute_lower_curvature()
    return G

def divide_group1(G: nx.Graph):
    """
    A function to divide edges whether they are across communities or within community edges.
    
    Parameters
    ----------
    G: A Networkx graph.
    
    Returns
    -------
    G: A Networkx graph with edge classification saved in edge attribute "group".
        group == 1: Within group community
        group == 2: Across group community
    """
    blockn = nx.get_node_attributes(G, "block")
    for e in G.edges(data=True):
        if blockn[e[0]] == blockn[e[1]]:
            e[2]["group"] = 0 
        else:
            e[2]["group"] = 1 
    return G

def bfc2(G: nx.Graph):
    """
    A function to calculate and save LRC value for every edge in a graph.
    
    Parameters
    ----------
    G: A Networkx graph.
    
    Returns
    -------
    G: A Networkx graph with lrc value saved in each edge attribute. 
    """
    bfc = BalancedForman(G)
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
    G = olliriccicurv(G)
    G = lowercurv(G)
    G = bfc2(G)
    G = divide_group1(G)
    edgelist = nx.to_pandas_edgelist(G)
    
    edgelist.to_csv('data/edgelist.csv', index=False)