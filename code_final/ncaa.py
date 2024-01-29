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
from lower_improve import LowerORicci
import community_eval2 as ceval

def lowercurv(G: nx.Graph):
    lrc = LowerORicci(G)
    G = lrc.lower_curvature()
    return G

def remove_curv(k, G, name):
    ebunch = []
    for e in G.edges(data=True):
        if e[2][name] < k:
            ebunch.append((e[0], e[1]))
    G.remove_edges_from(ebunch)

    
def group_nodes_by_feature(graph, feature):
    grouped_nodes = {}
    for node in graph.nodes():
        if feature in graph.nodes[node]:
            category = graph.nodes[node][feature]
            if category not in grouped_nodes:
                grouped_nodes[category] = [node]
            else:
                grouped_nodes[category].append(node)
    return grouped_nodes    
    
### READ IN NETWORK FILE ###
t0 = time.time()
G = nx.read_gml("/work/users/y/j/yjinpark/ncaa/football.gml", label = 'id')
G = lowercurv(G)
df = nx.to_pandas_edgelist(G)
grouped_by_category = group_nodes_by_feature(G, 'value')
groundtruth = []
for category, nodes in grouped_by_category.items():
    groundtruth.append(nodes)
gt_set = [set(inner_list) for inner_list in groundtruth]

t1 = time.time()
print("read data:")
print(t1- t0)

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
plt.savefig('/work/users/y/j/yjinpark/DBLP/gausmix_2compo.pdf', bbox_inches='tight')
plt.clf()
t6 = time.time()
print("fitting gaussian mixture:")
print(t6-t5)

### ALGORITHM EVALUATION ###

t7 = time.time()
print("read ground truth:")


## Without removal

# Angel
angel_coms = algorithms.label_propagation(G)
angel_set = [set(x) for x in angel_coms.communities]
angel_score_df = ceval.all_eval(gt_set, angel_set)

t8 = time.time()
print("lp done:")
print(t8-t7) 

# Multicom
multicom_coms = algorithms.leiden(G)
multicom_set = [set(x) for x in multicom_coms.communities]
multicom_score_df = ceval.all_eval(gt_set, multicom_set)

t9 = time.time()
print("leiden done:")
print(t9-t8) 


# Walkscan
walk_coms = algorithms.girvan_newman(G, level=10)
walk_set = [set(x) for x in walk_coms.communities]
walk_score_df = ceval.all_eval(gt_set, walk_set)

t10 = time.time()
print("gn done:")
print(t10-t9) 

t11 = time.time()
# Node Perception


t12 = time.time()
print("node done:")
print(t12-t11) 

# Ego
ego_coms = algorithms.walktrap(G)
ego_set = [set(x) for x in ego_coms.communities]
ego_score_df = ceval.all_eval(gt_set, ego_set)

t13 = time.time()
print("walk done:")
print(t13-t12) 





# Rename the 'Score' column to the name of the algorithm
angel_score_df.rename(columns={'Score': 'lp'}, inplace=True)
multicom_score_df.rename(columns={'Score': 'leiden'}, inplace=True)
walk_score_df.rename(columns={'Score': 'gn'}, inplace=True)
ego_score_df.rename(columns={'Score': 'walk'}, inplace=True)


# Concatenate all the DataFrames along the columns
combined_score_before = pd.concat([
    angel_score_df, multicom_score_df, walk_score_df, ego_score_df
], axis=1)


combined_score_before.to_csv("ncaa_before_fast.txt", sep='\t', index=True)



## With removal
remove_curv(x_val, G, "low")
t7 = time.time()
print("remove edges done:")
print(t7-t13)


# Angel
angel_coms = algorithms.label_propagation(G)
angel_set = [set(x) for x in angel_coms.communities]
angel_score_df = ceval.all_eval(gt_set, angel_set)

t8 = time.time()
print("lp done:")
print(t8-t7) 

# Multicom
multicom_coms = algorithms.leiden(G)
multicom_set = [set(x) for x in multicom_coms.communities]
multicom_score_df = ceval.all_eval(gt_set, multicom_set)

t9 = time.time()
print("leiden done:")
print(t9-t8) 


# Walkscan
walk_coms = algorithms.girvan_newman(G, level=10)
walk_set = [set(x) for x in walk_coms.communities]
walk_score_df = ceval.all_eval(gt_set, walk_set)

t10 = time.time()
print("gn done:")
print(t10-t9) 

t11 = time.time()
# Node Perception


t12 = time.time()
print("node done:")
print(t12-t11) 

# Ego
ego_coms = algorithms.walktrap(G)
ego_set = [set(x) for x in ego_coms.communities]
ego_score_df = ceval.all_eval(gt_set, ego_set)

t13 = time.time()
print("walk done:")
print(t13-t12) 





# Rename the 'Score' column to the name of the algorithm
angel_score_df.rename(columns={'Score': 'lp'}, inplace=True)
multicom_score_df.rename(columns={'Score': 'leiden'}, inplace=True)
walk_score_df.rename(columns={'Score': 'gn'}, inplace=True)
ego_score_df.rename(columns={'Score': 'walk'}, inplace=True)


# Concatenate all the DataFrames along the columns
combined_score_after = pd.concat([
    angel_score_df, multicom_score_df, walk_score_df, ego_score_df
], axis=1)
combined_score_after.to_csv("ncaa_after_fast.txt", sep='\t', index=True)

