# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:39:38 2021

@author: huafei huang
"""
# motif count
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import scipy.io as sio
import argparse

def caculate_VFlag(Na, Nb, inse, node_num):
    """
    Caculate VFlag tag vector using to 4-order motif counting.
    
    Input:
        Na:         (list) Node a neighbor 
        Nb:         (list) Node b neighbor
        inse:       (list) The intersection of Na and Nb
        node_num:   (int) Graph node number
    Output:
        VFlag:      (array) VFLAG vector
    """
    
    VFlag = np.zeros(node_num, dtype = int)
    for node in Na:
        VFlag[node] = 1 # 1-hop
    for node in Nb:
        VFlag[node] = 2 # 2-hop
    for node in inse:
        VFlag[node] = 3 # 1/2-hop

    return VFlag

def motiffeature(g, Sparse = False):
    """
    Calculate how many different kinds of motifs each node is in

    Input:
        g:              (networkx graph) input graph data
        Sparse:         (bool) Output matrix Sparse or not
    Output:
        motif_feature:  (matrix / coo_matix) motif feature matrix
    """

    # Initialize the motif feature dictionary
    node_num, node_list = g.number_of_nodes(), g.nodes()
    nm_dict = {}
    for node in node_list:
        nm_dict[node] = np.zeros(5, float)
    degree = dict(nx.degree(g))

    for node_a in node_list:
        if (node_a+1)%200==0:
            print('node count:',(node_a+1))
        Na = list(g.neighbors(node_a))
        for node_b in Na:
            if node_b < node_a:# can be same node
                continue
            Nb = list(g.neighbors(node_b))
            inse = list(set(Na).intersection(set(Nb)))

            # ensure there are no self-loop
            # M31 (Three-order triangle motif) Counting
            for node_c in inse:
                nm_dict[node_a][0] += 1/3
                nm_dict[node_b][0] += 1/3
                nm_dict[node_c][0] += 1/3


            # Get VFlag and M32 (Three-order path motif) Counting
            VFlag = caculate_VFlag(Na, Nb, inse, node_num)
            VFlag[node_a] = 0
            VFlag[node_b] = 0
            for i in range(0, len(VFlag)):
                if VFlag[i] == 1 or VFlag[i] == 2:
                    # M32
                    nm_dict[node_a][1] += 1/2
                    nm_dict[node_b][1] += 1/2
                    nm_dict[i][1]      += 1/2


            # M41 (Four-order fully connected motif) & M42 (Four-order stringed ring motif) Counting
            for node_c in inse:
                Nc = list(g.neighbors(node_c))
                for node_d in Nc:
                    if VFlag[node_d] == 3:
                        # M41
                        nm_dict[node_a][2] += 1/12
                        nm_dict[node_b][2] += 1/12
                        nm_dict[node_c][2] += 1/12
                        nm_dict[node_d][2] += 1/12
                    elif VFlag[node_d] == 2 or VFlag[node_d] == 1:
                        # M42
                        nm_dict[node_a][3] += 1/4
                        nm_dict[node_b][3] += 1/4
                        nm_dict[node_c][3] += 1/4
                        nm_dict[node_d][3] += 1/4


            # M43 (Four-order Square Motif) Counting 
            for node_c in Na:
                if VFlag[node_c] != 1 or node_c == node_b:
                    continue
                Nc = list(g.neighbors(node_c))
                for node_d in Nc:
                    if VFlag[node_d] == 2 and node_d != node_a:
                        nm_dict[node_a][4] += 1/4
                        nm_dict[node_b][4] += 1/4
                        nm_dict[node_c][4] += 1/4
                        nm_dict[node_d][4] += 1/4

    # Change dictionary to Matrix
    motif_feature = []
    for node in node_list:
        temp = [degree[node]]
        temp.extend(list(nm_dict[node]))
        motif_feature.append(temp)

    
    motif_feature = np.matrix(motif_feature)

    return motif_feature

def load_data(data_source):
    data = sio.loadmat("anomaly_data/{}.mat".format(data_source))
    print(data.keys())
    network = data['Network']
    labels=data['Label'] # anomaly
    attributes=data['Attributes']
    classes=data['Class']

    return network, attributes, labels,classes
    
# cora_both citationv1_both dblpv7_both acmv9_both pubmed_both
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', choices=['acmv9','citationv1','cora','dblpv7','pubmed'], help='dataset name: cora_both/citationv1_both/dblpv7_both...')
    args = parser.parse_args()
    data_source=args.dataset+'_both'
    adj, features, labels, classes=load_data(data_source)
    print('data:',data_source)
    g=nx.from_scipy_sparse_matrix(adj)
    #print(">> Motif counting...")
    # labels = ['degree', 'M31', 'M32', 'M41', 'M42', 'M43']
    print(">> Motif counting...")
    # labels = ['degree', 'M31', 'M32', 'M41', 'M42', 'M43']
    motif_feature = motiffeature(g)

    print(">> Motif Done...")

    scipy.io.savemat("data_motif/{}.mat".format(data_source+'_motif'), {'Network': adj, 'Label': labels,'Attributes':features,'Class':classes,'Motif':motif_feature})





