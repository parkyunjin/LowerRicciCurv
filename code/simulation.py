import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from zipfile import ZipFile
import os
import sys
sys.path.insert(1, "curvatures")
from balancedforman_curvature import BalancedForman
from lower_ricci_curvature import LowerORicci
from forman_ricci_curvature import FormanRicci


############################ Curvature calculating functions ##################################

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

def formancurv(G: nx.Graph):
    frc = FormanRicci(G)
    G = frc.compute_forman_curvature()
    return G

def balancedformancurv(G: nx.Graph):
    bfc = BalancedForman(G)
    G = bfc.compute_balancedformancurv()
    return G

def lowercurv(G: nx.Graph):
    lrc = LowerORicci(G)
    G = lrc.compute_lower_curvature()
    return G

def calculate_all_curv(p1, p2, G: nx.Graph, oalpha = 0.5, overbose="TRACE"):
    timedic = {}
    t0 = 0
    t1 = 0 
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0
    t0 = time.time()
    G = olliriccicurv(G)
    t1 = time.time()
    G = lowercurv(G)
    t2 = time.time()
    G = formancurv(G)
    t3 = time.time()
    G = balancedformancurv(G)
    t4 = time.time()
    timedic = {"p1":p1, "p2":p2, "orc": t1-t0, "lrc": t2-t1, "frc": t3-t2, "bfc": t4-t3}
    return G, timedic

##################################################################################################

def divide_group(G: nx.Graph):
    """Make two different dataframes for edges within a community and across communities (only for two layers).
    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    Returns
    -------
    df_within: Pandas dataframe for within community edges.
    df_across: Pandas dataframe for across communities edges.
    """
    blockn = nx.get_node_attributes(G, "block")
    for e in G.edges(data=True):
        if blockn[e[0]] == blockn[e[1]]:
            e[2]["group"] = 1 #"within a community"
        else:
            e[2]["group"] = 0 #"across communities"
    return G

######################SCORES#####################################################################

def scorecal(edgelist, curvaturetype): #proportion of within that is less than max(across)
    df = edgelist
    df1 = df[df["group"] == 1] #within community
    df1 = df1[curvaturetype].to_numpy()
    df2 = df[df["group"] == 0] #across community
    df2 = df2[curvaturetype].to_numpy()
    d = 0 
    total = df1.size
    if df1.size*df2.size != 0:
        for i in df1:
            if i <= np.nanmax(df2):
                d = d + 1
        result = d/total
    else:
        result = None
        
    return result


def curvature_diff(edgelist, curvaturetype):
    df = edgelist
    df1 = df[df["group"] == 1]
    df1 = df1[curvaturetype].to_numpy()
    df2 = df[df["group"] == 0]
    df2 = df2[curvaturetype].to_numpy()
    if df1.size*df2.size != 0:
        d1 = np.nanmin(df1) - np.nanmax(df2)
    else: 
        d1 = None
    return d1

def proportioncal(edgelist, curvaturetype):
    #this cannot be calculated in a network, calculated by repeat
    orc_diff = curvature_diff(edgelist, "ricciCurvature")
    lrc_diff = curvature_diff(edgelist, "lowerCurvature")
    frc_diff = curvature_diff(edgelist, "formanCurvature")
    bfc_diff = curvature_diff(edgelist, "balancedforman")
    if orc_diff>0:
        orc_diff_p = orc_diff_p + 1
    if lrc_diff>0:
        lrc_diff_p = lrc_diff_p + 1
    if frc_diff>0:
        frc_diff_p = frc_diff_p + 1
    if bfc_diff>0:
        bfc_diff_p = bfc_diff_p + 1
    return orc_diff_p, lrc_diff_p, frc_diff_p, bfc_diff_p





def quantilescore(edgelist, curvaturetype):
    df = edgelist
    df1 = df[df["group"] == 1]
    df2 = df[df["group"] == 0]
    df1 = df1[curvaturetype].to_numpy()
    df2 = df2[curvaturetype].to_numpy()
    if df1.size*df2.size != 0:
        across_max = np.nanmax(df2)
        within_min = np.nanmin(df1)
        a_max_percentage = len(df1[df1 < across_max])/len(df1)
        w_min_percentage = len(df2[df2 > within_min])/len(df2)
        result = w_min_percentage + a_max_percentage
    else:
        result = None
    return result


###################################################################


def mainf(parent, p1, p2, n1=50, n2=50, T=100):
    dftime = pd.DataFrame(columns=["p1", "p2", "orc", "lrc", "frc", "bfc"])
    diclist = []
    results = []
    
    #default value for gap >0
    orc_diff = -1

    lrc_diff = -1
    frc_diff = -1
    bfc_diff = -1
    
    orc_diff_p = 0
    lrc_diff_p = 0
    frc_diff_p = 0
    bfc_diff_p = 0
    
    #default value for score1
    orc_score1 = 0
    lrc_score1 = 0
    frc_score1 = 0
    bfc_score1 = 0
    
    #default value for score2
    orcq1 = 0
    lrcq1 = 0
    frcq1 = 0 
    bfcq1 = 0 
    
    #timelist = []
    parent_dir = parent
    if p1 >= p2:
        sizes = [n1, n2]
        prob = [[p1, p2], [p2, p1]]
        for t in range(T): #repeat for T times
            title = ""
            path = ""
            directory = ""
            g = nx.stochastic_block_model(sizes, prob, seed=t) #make one-level SBM
            #g.remove_nodes_from(list(nx.isolates(g)))
            g, timedic = calculate_all_curv(p1, p2, g) #calculate curvatures
            dfnewtime = pd.DataFrame.from_dict([timedic])
            dftime = pd.concat([dftime, dfnewtime]) #save time spent

            g = divide_group(g) #divide whether the edge is within or across
            edgelist = nx.to_pandas_edgelist(g)
            
            #save edgelist for reproduction
            edgelist2 = edgelist[["source","target", "group"]]
            directory = str(t//10)
            path = os.path.join(parent_dir, directory)
            if not os.path.exists(path):
                os.mkdir(path)
            title = path+ "/" + str(t) + ".txt" 
            edgelist2.to_csv(title, sep='\t', index=False)
            
            #gap>0 calculation
            orc_diff = curvature_diff(edgelist, "ricciCurvature")
            lrc_diff = curvature_diff(edgelist, "lowerCurvature")
            frc_diff = curvature_diff(edgelist, "formanCurvature")
            bfc_diff = curvature_diff(edgelist, "balancedforman")
            if orc_diff != None:
                if orc_diff>0:
                    orc_diff_p = orc_diff_p + 1
            if lrc_diff != None:
                if lrc_diff>0:
                    lrc_diff_p = lrc_diff_p + 1
            if frc_diff != None:
                if frc_diff>0:
                    frc_diff_p = frc_diff_p + 1
            if bfc_diff != None:
                if bfc_diff>0:
                    bfc_diff_p = bfc_diff_p + 1

                
            #score1 calculation
            orc_s1 = scorecal(edgelist, "ricciCurvature")
            lrc_s1 = scorecal(edgelist, "lowerCurvature")
            frc_s1 = scorecal(edgelist, "formanCurvature")
            bfc_s1 = scorecal(edgelist, "balancedforman")

            if orc_s1 != None:
                orc_score1 = orc_score1 + orc_s1
            if lrc_s1 != None:
                lrc_score1 = lrc_score1 + lrc_s1
            if frc_s1 != None:
                frc_score1 = frc_score1 + frc_s1
            if bfc_s1 != None:
                bfc_score1 = bfc_score1 + bfc_s1
           
            
            #score2 calculation
            orq1= quantilescore(edgelist, "ricciCurvature")
            lrq1 = quantilescore(edgelist, "lowerCurvature")
            frq1 = quantilescore(edgelist, "formanCurvature")
            bfq1 = quantilescore(edgelist, "balancedforman")

            if orq1 != None:
                orcq1 = orcq1 + orq1
            if lrq1 != None:
                lrcq1 = lrcq1 + lrq1
            if frq1 != None:
                frcq1 = frcq1 + frq1
            if bfq1 != None:
                bfcq1 = bfcq1 + bfq1

            
            
        #final gap>0 proportion    
        orc_diff_p = orc_diff_p/T
        lrc_diff_p = lrc_diff_p/T
        frc_diff_p = frc_diff_p/T
        bfc_diff_p = bfc_diff_p/T
        #final score1 mean    
        orc_score1 = orc_score1/T
        lrc_score1 = lrc_score1/T
        frc_score1 = frc_score1/T
        bfc_score1 = bfc_score1/T
        #final score2 mean
        orcq1 = orcq1/T 
        lrcq1 = lrcq1/T 
        frcq1 = frcq1/T 
        bfcq1 = bfcq1/T 
        
        
        result1 = {"p1": p1, "p2": p2, "n1": n1, "n2": n2, "orc_gap": orc_diff_p, "lower_gap": lrc_diff_p, "frc_gap": frc_diff_p, 
            "bfc_gap": bfc_diff_p, "orcs": orc_score1, "lrcs": lrc_score1, "frcs": frc_score1, "bfcs": bfc_score1, "orcq": orcq1, "lrcq": lrcq1, "frcq": frcq1, "bfcq": bfcq1}
    else:
        print("p1 should be greater than p2")
    return result1, dftime
                    
                    
                    
                    
def write_csv_sbm(diclist, title):
    fields = ["p1", "p2", "n1", "n2", "orc_gap", "lower_gap", "frc_gap", 
            "bfc_gap", "orcs", "lrcs", "frcs", "bfcs", "orcq", "lrcq", "frcq", "bfcq"] 
    title = title + ".csv"
    with open(title, 'a', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader()
        writer.writerows(diclist)


if __name__ == "__main__":
    #You need to have one_sum folder to save this result#
    numberstr = "01_1"
    listname = "/input/part" + numberstr + ".txt"
    with open(listname, "r") as file:
        Lines = file.readlines()
    number = 0
    diclist = []
    parent_dir = "one_sum"
    middled = "newoutput" + numberstr
    path0 = os.path.join(parent_dir, middled)
    if not os.path.exists(path0):
                os.mkdir(path0)
    for line in Lines:
        data_into_list = line.replace('\n', '').split("\t")
        y = [float(x) for x in data_into_list]
        
        parentd = "result"+ str(number)
        path1 = os.path.join(path0, parentd)
        if not os.path.exists(path1):
                os.mkdir(path1)
        
        dic, timedf = mainf(p1 = y[0], p2 = y[1], n1 = 50, n2 = 50, T= 100, parent = path1)
        
        diclist.append(dic)
        

        number = number + 1
    
    summarypath = "/one_sum/summary" +numberstr
    write_csv_sbm(diclist, summarypath)
    print(timedf.sum())
    print(number)
    
  
