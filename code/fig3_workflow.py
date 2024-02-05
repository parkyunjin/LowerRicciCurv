import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import norm
from scipy.sparse import save_npz
import random

def remove_curv(k, G, name):
    """
    A function to remove edges that have curvature values that are less than certain criteria.

    Parameters
    ----------
    k: Certain curvature value that becomes the criteria.
    G: A Networkx graph.
    name: Name of the curvature. (Possible values: "lrc", "frc", "bfc", "ricciCurvature")
    """
    ebunch = []
    for e in G.edges(data=True):
        if e[2][name] < k:
            ebunch.append((e[0], e[1]))
    G.remove_edges_from(ebunch)

if __name__ == "__main__":
    
    ### READ GRAPH DATA ###
    df = pd.read_csv('/data/edgelist.csv') # Read SBM edgelist
    G = nx.from_pandas_edgelist(df, edge_attr=True)
    pos = nx.spring_layout(G, seed = 0) # Set the layout of the graph
    lowdf = df["low"] # Dataframe consisting of LRC

    
    ### GMM & FIND ALPHA ###
    # create GMM model object with two components
    x = np.ravel(lowdf).astype(float)
    x = x.reshape(-1, 1)
    gmm2 = GMM(n_components = 2, max_iter=1000, random_state=10, covariance_type = 'full')

    # find useful parameters
    mean2 = gmm2.fit(x).means_
    covs2  = gmm2.fit(x).covariances_
    weights2 = gmm2.fit(x).weights_

    # find the middle point
    p = 100 #the probability of x, default value
    x_val = -3 #x value, default value

    sgmm = min(mean2[0][0], mean2[1][0])
    bgmm = max(mean2[0][0], mean2[1][0])

    for i in np.arange(round(float(sgmm), 2), round(float(bgmm),2)+0.01, 0.01):
        y_0 = norm.pdf(i, float(mean2[0][0]), np.sqrt(float(covs2[0][0][0])))*weights2[0] # 1st gaussian
        y_1 = norm.pdf(i, float(mean2[1][0]), np.sqrt(float(covs2[1][0][0])))*weights2[1] # 2nd gaussian
        p_new = y_0 + y_1
        if p > p_new:
            p = p_new
            x_val = i
        else:
            continue
    print("middle point:")
    print(x_val)
    print("probability of the middle point:")
    print(p)

    # create necessary things to plot
    x_axis = np.arange(-2.5, 2.5, 0.1)
    y_axis0 = norm.pdf(x_axis, float(mean2[0][0]), np.sqrt(float(covs2[0][0][0])))*weights2[0] # 1st gaussian
    y_axis1 = norm.pdf(x_axis, float(mean2[1][0]), np.sqrt(float(covs2[1][0][0])))*weights2[1] # 2nd gaussian

    # Choose a colormap
    cmap = plt.cm.viridis
    edge_values = {(i,j): G[i][j]["low"] for i, j in G.edges()} 

    # Manually set the range for your colormap
    vmin = min(edge_values.values())
    vmax = max(edge_values.values())

    
    ### DRAW THE FIGURE ### 
    fig = plt.figure(figsize=(24, 6))
    node_size = 50
    fixed_positions = nx.spring_layout(G, seed=0)

    initial_edge_colors = {e: cmap((edge_values[e] - vmin) / (vmax - vmin)) for e in G.edges()}
    
    # Fig 3(a)
    ax1 = fig.add_subplot(1, 4, 1)
    pos = nx.spring_layout(G, seed = 0)
    ec = nx.draw_networkx_edges(G, fixed_positions, edge_color=[initial_edge_colors[e] for e in G.edges() if e in initial_edge_colors])
    nc = nx.draw_networkx_nodes(G, fixed_positions, node_color='black', node_size=node_size)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), ax=ax1, orientation='vertical')
    cbar.set_label('Lower-Ricci Curvature')

    ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes, fontsize=16, va='top', ha='right')

    # Fig 3(b)
    ax2 = fig.add_subplot(1, 4, 2)
    n, bins, patches = plt.hist(edge_values.values(), bins=30, edgecolor = "black")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cmap((c - vmin) / (vmax - vmin)))

    ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes, fontsize=16, va='top', ha='right')

    # Fig 3(c)
    ax3 = fig.add_subplot(1, 4, 3)
    n, bins, patches = plt.hist(edge_values.values(), bins=30, edgecolor = "black", density = True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cmap((c - vmin) / (vmax - vmin)))
    plt.plot(x_axis, y_axis0, lw=3, color = "C0", label = "GMM1")
    plt.plot(x_axis, y_axis1, lw=3, color = "C1", label = "GMM2")
    plt.plot(x_axis, y_axis0+y_axis1, lw=3, c="C2", ls='dashed', label = "GMM")
    plt.xlim(-2.5, 2.5)
    plt.xlabel(r"LRC", fontsize=10)
    plt.ylabel(r"Density", fontsize=10)
    
    # Draw alpha in the plot
    plt.axvline(x=x_val, color='red', linestyle='--', label=r'$\beta$')
    plt.legend(loc='upper right')

    ax3.text(-0.1, 1.1, '(c)', transform=ax3.transAxes, fontsize=16, va='top', ha='right')

    # Fig 3(d)
    remove_curv(x_val, G, "low")
    ax4 = fig.add_subplot(1, 4, 4)
    pos = nx.spring_layout(G, seed = 0)
    ec = nx.draw_networkx_edges(G, fixed_positions, edge_color=[initial_edge_colors[e] for e in G.edges() if e in initial_edge_colors])
    nc = nx.draw_networkx_nodes(G, fixed_positions, node_color='black', node_size=node_size)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), ax=ax4, orientation='vertical')
    cbar.set_label('Lower-Ricci Curvature')
    ax4.text(-0.1, 1.1, '(d)', transform=ax4.transAxes, fontsize=16, va='top', ha='right')
    plt.tight_layout()

    
    ### SAVE THE FIGURE INTO PDF ###
    file_path = 'workflow.pdf'
    fig.savefig(file_path)