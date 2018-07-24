import os
import sys

sys.path.insert(1, '../src')

import numpy as np
import networkx as nx


def generateNXGraphLite(adj_mat):
    x = nx.from_numpy_matrix(adj_mat)
    return x

if len(sys.argv)!=2:
        raise Exception('This function must be called with one parameter. The full path (either relative or absolute) to the file containing the frequent subgraphs.')

file_path, file_name = os.path.split(sys.argv[1])

with open(os.path.join(file_path,file_name)) as f:
    graphs = []
    supports = []
    for l in f:
        #Skip empty lines
        if l == '\n': 
            continue
        #If its a new graph. Store the current one and initialize the next
        if l[0] == '-':
            graphs.append(generateNXGraphLite(adj_mat))
            supports.append(support)
        if l[0] == 't':
            #Initialize next graph
            support = 0
            first_edge = True
            vertices = []
        #Its a new vertex. Add it.
        if l[0] == 'v':
            vertices.append((int(l.split(' ')[1])))
        #Its a new edge. Add it.
        if l[0] == 'e':
            if first_edge:
                adj_mat = np.zeros((len(vertices),len(vertices)))
                first_edge = False
            #Add edge
            adj_mat[int(l.split(' ')[1]),int(l.split(' ')[2])]+=1
        #Its the graph support. Store it.
        if l[0] == 'S':
            support = int(l.split(' ')[1])

#Order graphs by support
graphs = [x for _,x in sorted(zip(supports,graphs), reverse=True)]
supports = sorted(supports, reverse=True)

#Store all subgraphs in a file
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(os.path.join(file_path,os.path.splitext(file_name)[0]+'.pdf')) as pdf:
    for graph,support in zip(graphs,supports):
        plt.figure(figsize=(7,7))
        plt.axis('off')
        edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
        pos = nx.spring_layout(graph,k=0.5,iterations=30)
        nx.draw(graph, pos, edgelist=edges, edge_color=weights,
                node_size = 700, width=5.0, edge_cmap=plt.cm.winter)
        #nx.draw_networkx_labels(graph, pos, labels=labels, font_size = 20)
        plt.title('Support: '+str(support))
        pdf.savefig()
        plt.close()

