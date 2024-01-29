import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.sparse import save_npz
import random

df = pd.read_csv('/data/edgelist.csv')
G = nx.from_pandas_edgelist(df, edge_attr=True)


##Draw ORC##
# Choose a colormap
cmap = plt.cm.viridis
edge_values = {(i,j): G[i][j]["ricciCurvature"] for i, j in G.edges()} 

# Manually set the range for your colormap
vmin = min(edge_values.values())
vmax = max(edge_values.values())

node_size = 50

fig = plt.figure(figsize=(12, 6))

# Draw the graph
ax1 = fig.add_subplot(1, 2, 1)
pos = nx.spring_layout(G, seed = 0)
ec = nx.draw_networkx_edges(G, pos, edge_color=[cmap((value - vmin) / (vmax - vmin)) for value in edge_values.values()])
nc = nx.draw_networkx_nodes(G, pos, node_color='black', node_size=node_size)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), ax=ax1, orientation='vertical')
cbar.set_label('Ollivier-Ricci Curvature')

ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes, fontsize=16, va='top', ha='right')

# Draw the histogram
ax2 = fig.add_subplot(1, 2, 2)
n, bins, patches = plt.hist(edge_values.values(), bins=30, edgecolor = "black")
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for c, p in zip(bin_centers, patches):
    plt.setp(p, 'facecolor', cmap((c - vmin) / (vmax - vmin)))
plt.tight_layout()
ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes, fontsize=16, va='top', ha='right')

# Save the figure
file_path = 'orc.pdf'
fig.savefig(file_path)


##Draw LRC##
df1 = df[df["group"] == 0]
df2 = df[df["group"] == 1]
# Choose a colormap
cmap = plt.cm.viridis
edge_values = {(i,j): G[i][j]["low"] for i, j in G.edges()} 

# Manually set the range for your colormap
vmin = min(edge_values.values())
vmax = max(edge_values.values())

edge_valuesb = {(i,j): G[i][j]["bfcsecond"] for i, j in G.edges()} 
vminb = min(edge_valuesb.values())
vmaxb = max(edge_valuesb.values())

fig = plt.figure(figsize=(18, 6))

# Draw the graph
ax1 = fig.add_subplot(1, 3, 1)
pos = nx.spring_layout(G, seed = 0)
ec = nx.draw_networkx_edges(G, pos, edge_color=[cmap((value - vmin) / (vmax - vmin)) for value in edge_values.values()])
nc = nx.draw_networkx_nodes(G, pos, node_color='black', node_size=node_size)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), ax=ax1, orientation='vertical')
cbar.set_label('Lower-Ricci Curvature')

ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes, fontsize=16, va='top', ha='right')

# Draw the histogram
ax2 = fig.add_subplot(1, 3, 2)
n, bins, patches = plt.hist(edge_values.values(), bins=30, edgecolor = "black")
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for c, p in zip(bin_centers, patches):
    plt.setp(p, 'facecolor', cmap((c - vmin) / (vmax - vmin)))

ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes, fontsize=16, va='top', ha='right')


ax3 = fig.add_subplot(1, 3, 3)
plt.hist(df1["bfcsecond"], alpha=0.9, label="within a community", color = "#86d548", edgecolor = "black")
plt.hist(df2["bfcsecond"], alpha=1, label="across communities", color = "#481f70",  edgecolor = "black")
plt.legend(loc='upper right')
plt.tight_layout()
ax3.text(-0.1, 1.1, '(c)', transform=ax3.transAxes, fontsize=16, va='top', ha='right')

# Save the figure
file_path = 'lrc_bfc.pdf'
fig.savefig(file_path)

file_path