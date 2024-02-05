import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.sparse import save_npz
import random

if __name__ == "__main__":
    
    ### READ GRAPH DATA ###
    df = pd.read_csv('/data/edgelist.csv') # Read SBM edgelist
    G = nx.from_pandas_edgelist(df, edge_attr=True)
    pos = nx.spring_layout(G, seed = 0) # Set the layout of the graph


    # Choose a colormap
    cmap = plt.cm.viridis
    edge_values = {(i,j): G[i][j]["ricciCurvature"] for i, j in G.edges()} 
    node_size = 50

    # Manually set the range for your colormap
    vmin = min(edge_values.values())
    vmax = max(edge_values.values())

    fig = plt.figure(figsize=(18, 6))

    # Fig 1(a)
    ax1 = fig.add_subplot(1, 3, 1)
    pos = nx.spring_layout(G, seed = 0)
    ec = nx.draw_networkx_edges(G, pos, edge_color=[cmap((value - vmin) / (vmax - vmin)) for value in edge_values.values()])
    nc = nx.draw_networkx_nodes(G, pos, node_color='black', node_size=node_size)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), ax=ax1, orientation='vertical')
    cbar.set_label('Lower-Ricci Curvature')

    ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes, fontsize=16, va='top', ha='right')

    # Fig 1(b)
    ax2 = fig.add_subplot(1, 3, 2)
    n, bins, patches = plt.hist(edge_values.values(), bins=30, edgecolor = "black")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cmap((c - vmin) / (vmax - vmin)))

    ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes, fontsize=16, va='top', ha='right')


    # Save the figure
    file_path = 'orc2.pdf'
    fig.savefig(file_path)
