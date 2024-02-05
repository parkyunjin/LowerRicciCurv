import networkx as nx
import math
import numpy as np

                
class LowerORicci:
    def __init__(self, G: nx.Graph):
        """
        A class to compute the Lower-Ricci curvature (LRC) of a given NetworkX graph.
        Current version can only compute LRC for unweighted and undirected graph.
        
        Parameters
        ----------
        G: NetworkX graph.
        """
        self.G = G.copy()
    
    
    def compute_lower_curvature(self):
        """Compute LRC for all edges in G.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "lrc" on edges.
        """
        C = {}
        for i, j in self.G.edges:
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))
            n_max = max(n_i, n_j)
            n_min = min(n_i, n_j)
            C[(i, j)] = 2/n_i + 2/n_j - 2 + 2*n_ij/n_max + n_ij/n_min
        nx.set_edge_attributes(self.G, C, "lrc")
        return self.G