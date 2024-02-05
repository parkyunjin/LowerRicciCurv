import networkx as nx
import time
import math
import sys
sys.path.insert(1, '/work/users/y/j/yjinpark/curvature_community/functions')
from balancedforman_numba import balanced_forman_curvature, bfc_two

class BalancedForman:
    def __init__(self, G: nx.Graph):
        """A class to compute Balanced Forman curvature (BFC) of a given NetworkX graph.

        Parameters
        ----------
        G : NetworkX graph.
        """

        self.G = G.copy()
        

    #saving the result in a dictionary
    def set_balancedforman_edge(self,C):
        """Make a dictionary called attri that saves the value of BFC for each edge
        Parameters
        ----------
        C: The curvature matrix.
        
        Returns
        -------
        attri: A dictionary that saves that value of BFC as value and edge as keys. 
        """
        attri = {}
        nodelist = list(self.G)
        for i, j in nx.edges(self.G):
            x = nodelist.index(i)
            y = nodelist.index(j)
            attri[(i,j)] = C[x, y]
            attri[(j,i)] = C[y, x]
        return attri

    def compute_balancedformancurv(self):
        """Compute Balanced Forman Curvature of edges.
        
        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "nfc" on edges.
        """
        A = nx.to_numpy_array(self.G)
        curv = balanced_forman_curvature(A, C = None)
        a = self.set_balancedforman_edge(curv)
        nx.set_edge_attributes(self.G, a, "bfc")
        return self.G
    
    def compute_bfcsecond(self):
        """Compute Delta of Balanced Forman Curvature for all edges.
        Parameters
        ----------
        G : NetworkX graph
            A given directional or undirectional NetworkX graph.
        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "bfcsecond" on nodes and edges.
        """
        A = nx.to_numpy_array(self.G)
        curv = bfc_two(A, C = None)
        a = self.set_balancedforman_edge(curv)
        nx.set_edge_attributes(self.G, a, "bfcsecond")
        return self.G