import networkx as nx
import math
import numpy as np


class FormanRicci:
    def __init__(self, G: nx.Graph):
        """
        A class to compute the Forman-Ricci curvature (FRC) of a given NetworkX graph.
        Current version can only compute LRC for unweighted and undirected graph.
        
        Parameters
        ----------
        G: NetworkX graph.
        """
        self.G = G.copy()
    
    
    def forman_curvature(self):
        """Compute FRC for all edges in G.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "frc" on edges.
        """
        C = {}
        for i, j in self.G.edges:
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))
            C[(i, j)] = 4 - n_i + n_j 3*n_ij
        nx.set_edge_attributes(self.G, C, "frc")
        return self.G