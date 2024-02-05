import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mpl
from cdlib import algorithms
from cdlib import datasets
import time
import sys
sys.path.insert(1, "curvatures")
from lower_ricci_curvature import LowerORicci

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
 

if __name__ == '__main__':
    
    ### READ IN NETWORK FILE ###
    t0 = time.time()
    G, gt_coms = datasets.fetch_network_ground_truth("youtube","networkx") #"youtube" can be replaced to "dblp", "amazon"
    gnode = set(G.nodes())
    gt = set(gt_coms.to_node_community_map().keys())
    dnode = gnode-gt
    G.remove_nodes_from(dnode)
    t1 = time.time()
    print("read data:")
    print(t1- t0)
    gt_set = [set(x) for x in gt_coms.communities]

    ### CALCULATE CURVATURES ###
    t2 = time.time()
    G = lowercurv(G)
    t3 = time.time()
    print("calculate lower:")
    print(t3-t2)
    edgedf = nx.to_pandas_edgelist(G)
    lowdf = edgedf["low"]
    t4 = time.time()
    print("make low df:")
    print(t4-t3)


    ### HISTOGRAM & GMM & FIND ALPHA ###
    # create GMM model object with two components
    t5 = time.time()
    x = np.ravel(lowdf).astype(float)
    x = x.reshape(-1, 1)
    gmm2 = GMM(n_components = 2, max_iter=1000, random_state=10, covariance_type = 'full')

    # find useful parameters
    mean2 = gmm2.fit(x).means_
    covs2  = gmm2.fit(x).covariances_
    weights2 = gmm2.fit(x).weights_

    # find the middle point
    p = 100 #the probability of x
    x_val = -3

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

    # Plot histogram with GMM
    plt.hist(x, density=True, color='black', bins=30)
    plt.plot(x_axis, y_axis0, lw=3, c='C0')
    plt.plot(x_axis, y_axis1, lw=3, c='C1')
    plt.plot(x_axis, y_axis0+y_axis1, lw=3, c='C2', ls='dashed')
    plt.xlim(-2.5, 2.5)
    #plt.ylim(0.0, 2.0)
    plt.xlabel(r"LRC", fontsize=20)
    plt.ylabel(r"Density", fontsize=20)
    # Draw alpha in the plot
    plt.vlines(x=x_val, ymin=0, ymax=p, colors='red', ls='solid', lw=2, label='alpha')
    plt.show()
    plt.savefig('gausmix_2compo.pdf', bbox_inches='tight')
    plt.clf()
    t6 = time.time()
    print("fitting gaussian mixture:")
    print(t6-t5)

    ### ALGORITHM EVALUATION ###

    t7 = time.time()
    print("read ground truth:")


    ## Without removal ##

    # Angel
    angel_coms = algorithms.angel(G, threshold = 0.5)
    angel_set = [set(x) for x in angel_coms.communities]
    angel_score_df = ceval.all_eval(gt_set, angel_set)

    t8 = time.time()
    print("angel done:")
    print(t8-t7) 


    # Ego
    ego_coms = algorithms.ego_networks(G)
    ego_set = [set(x) for x in ego_coms.communities]
    ego_score_df = ceval.all_eval(gt_set, ego_set)

    t9 = time.time()
    print("ego done:")
    print(t9-t8) 


    # Kclique
    clique_coms = algorithms.kclique(G, k =3)
    clique_set = [set(x) for x in clique_coms.communities]
    clique_score_df = ceval.all_eval(gt_set, clique_set)

    t10 = time.time()
    print("kclique done:")
    print(t10-t9) 

    # SLPA
    slpa_coms = algorithms.slpa(G)
    slpa_set = [set(x) for x in slpa_coms.communities]
    slpa_score_df = ceval.all_eval(gt_set, slpa_set)

    t11 = time.time()
    print("SLPA done:")
    print(t11-t10) 


    # Rename the 'Score' column to the name of the algorithm
    angel_score_df.rename(columns={'Score': 'Angel'}, inplace=True)
    ego_score_df.rename(columns={'Score': 'Ego'}, inplace=True)
    clique_score_df.rename(columns={'Score': 'Kclique'}, inplace=True)
    slpa_score_df.rename(columns={'Score': 'SLPA'}, inplace=True)

    # Concatenate all the DataFrames along the columns
    combined_score_before = pd.concat([
    angel_score_df, ego_score_df, clique_score_df, slpa_score_df
    ], axis=1)


    combined_score_before.to_csv("youtube_before_fast.txt", sep='\t', index=True)



    ## With removal ##
    remove_curv(x_val, G, "low")
    t12 = time.time()
    print("remove edges done:")
    print(t12-t11)


    # Angel
    angel_coms = algorithms.angel(G, threshold = 0.5)
    angel_set = [set(x) for x in angel_coms.communities]
    angel_score_df = ceval.all_eval(gt_set, angel_set)

    t13 = time.time()
    print("angel done:")
    print(t13-t12) 


    # Ego
    ego_coms = algorithms.ego_networks(G)
    ego_set = [set(x) for x in ego_coms.communities]
    ego_score_df = ceval.all_eval(gt_set, ego_set)

    t14 = time.time()
    print("ego done:")
    print(t14-t13) 


    # Kclique
    clique_coms = algorithms.kclique(G, k =3)
    clique_set = [set(x) for x in clique_coms.communities]
    clique_score_df = ceval.all_eval(gt_set, clique_set)

    t15 = time.time()
    print("kclique done:")
    print(t15-t14) 

    # SLPA
    slpa_coms = algorithms.slpa(G)
    slpa_set = [set(x) for x in slpa_coms.communities]
    slpa_score_df = ceval.all_eval(gt_set, slpa_set)

    t16 = time.time()
    print("SLPA done:")
    print(t16-t15) 


    # Rename the 'Score' column to the name of the algorithm
    angel_score_df.rename(columns={'Score': 'Angel'}, inplace=True)
    ego_score_df.rename(columns={'Score': 'Ego'}, inplace=True)
    clique_score_df.rename(columns={'Score': 'Kclique'}, inplace=True)
    slpa_score_df.rename(columns={'Score': 'SLPA'}, inplace=True)

    # Concatenate all the DataFrames along the columns
    combined_score_after = pd.concat([
    angel_score_df, ego_score_df, clique_score_df, slpa_score_df
    ], axis=1)

    combined_score_after.to_csv("youtube_after_fast.txt", sep='\t', index=True)

