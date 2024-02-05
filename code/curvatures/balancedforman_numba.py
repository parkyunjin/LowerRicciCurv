# A modified version of numba.py in understanding-oversquashing package.
# Reference: 
# Topping, Jake, et al. "Understanding over-squashing and bottlenecks on graphs via curvature." arXiv preprint arXiv:2111.14522 (2021).

from numba import jit, prange
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)


@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    """
        A function to compute the Balanced Forman curvature (BFC) of a given NetworkX graph.
        Current version can only compute BFC for unweighted graph.
        
        Parameters
        ----------
        A: Adjacency matrix of a graph.
        A2: A squared.
        d_in: Row sum of A
        d_out: Column sum of A
        N: The total number of nodes in a graph.
        C: Curvature matrix that saves curvature of edge (ij) in C_ij
        """
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0: #if no edges, then bfc is zero
                C[i, j] = 0
                continue

            if d_in[i] > d_out[j]: #only for directed case
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                continue

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)
                

def balanced_forman_curvature(A, C=None):
    """
        A final function to compute the Balanced Forman curvature (BFC) of a given NetworkX graph.
        
        Parameters
        ----------
        A: Adjacency matrix of a graph.
        C: Curvature matrix that saves curvature of edge (ij) in C_ij. Default is n by n zero matrix. 
        
        Returns
        -------
        C: Curvature matrix that saves curvature of edge (ij) in C_ij
        """
    N = A.shape[0] 
    A2 = np.matmul(A, A) 
    d_in = A.sum(axis=0) 
    d_out = A.sum(axis=1) 
    if C is None:
        C = np.zeros((N, N)) 

    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C 


@jit(nopython=True)
def _bfc_second(A, A2, d_in, d_out, N, C):
    """
        A function to compute the Delta of Balanced Forman curvature (BFC) of a given NetworkX graph.
        Current version can only compute BFC for unweighted graph.
        
        Parameters
        ----------
        A: Adjacency matrix of a graph.
        A2: A squared.
        d_in: Row sum of A
        d_out: Column sum of A
        N: The total number of nodes in a graph.
        C: Curvature matrix that saves curvature of edge (ij) in C_ij
        """
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0: #if no edges, then bfc is zero
                C[i, j] = 0
                continue

            if d_in[i] > d_out[j]: #only for directed case
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                continue

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = 0
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)
                
def bfc_two(A, C=None):
    """
        A final function to compute the Delta of Balanced Forman curvature (BFC) of a given NetworkX graph.
        
        Parameters
        ----------
        A: Adjacency matrix of a graph.
        C: Curvature matrix that saves curvature of edge (ij) in C_ij. Default is n by n zero matrix. 
        
        Returns
        -------
        C: Curvature matrix that saves curvature of edge (ij) in C_ij
        """
    N = A.shape[0] 
    A2 = np.matmul(A, A) 
    d_in = A.sum(axis=0) 
    d_out = A.sum(axis=1) 
    if C is None:
        C = np.zeros((N, N)) 

    _bfc_second(A, A2, d_in, d_out, N, C)
    return C 
