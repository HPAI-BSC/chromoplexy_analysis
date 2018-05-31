import numpy as np
from graphnx import printGraph

def generateNXGraphLite(adj_mat):
    import networkx as nx
    x = nx.from_numpy_matrix(adj_mat)
    return x
file_location = "../results/gspan_1K_frequent_subgraphs_old.txt"


with open(file_location) as f:
    for l in f:
        #Skip empty lines
        if l == '\n': 
            continue
        #If its a new graph. Print the current one and initialize the next
        if l[0] == '-':
            #TODO print graph
            g = generateNXGraphLite(adj_mat)
            g_plot = printGraph(g, visualize = False)

        if l[0] == 't':
            #Initialize next graph
            support = 0
            first_edge = True
            vertices = []
        #Its a new vertex. Add it.
        if l[0] == 'v':
            vertices.append((int(l.split(' ')[1]),int(l.split(' ')[2])))
        #Its a new edge. Add it.
        if l[0] == 'e':
            if first_edge:
                adj_mat = np.zeros((len(vertices),len(vertices)))
                first_edge = False
            #Add edge
            adj_mat[int(l.split(' ')[1]),int(l.split(' ')[2])]
        #TODO
        #Its the graph support. Store it.
        if l[0] == 'S':
            support = int(l.split(' ')[1])
