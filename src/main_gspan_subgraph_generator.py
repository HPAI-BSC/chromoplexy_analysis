'''
Reads chromoplexy breaks, generates a graph using a sliding window of fixed size, and stores all the resulting connected subraphs in a file compatible with gSpan[1].
[1] 
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import generateNXGraph
import networkx as nx

import os
import sys

if len(sys.argv)!=2:
        raise Exception('This function must be called with one parameter. An integer indicating the length of the sliding window.')
max_distance = int(sys.argv[1])
subgraphs = []

#Directory containing the files
data_path = '../data/allfiles'
#Iterate over the files
for filename in os.listdir(data_path):
        breaks, list_of_pairs = load_breaks(os.path.join(data_path,filename))
        if len(breaks) == 0:
            #print 'WARNING: Empty data file',file_name
            continue
        adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance)
        g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, self_links=False, connected_only=True)
        candidates = list(nx.connected_component_subgraphs(g))
        for c in candidates:
            if len(c.nodes())>=3 and len(c.nodes())>=3:
                subgraphs.append(c)

#Iterate over subgraphs and store
with open('../results/gspan_subgraphs_w'+str(max_distance)+'.txt','w') as f:
    counter = 0
    for g in subgraphs:
        f.write('t # '+str(counter)+'\n')
        counter+=1
        #Iterate over vertices
        for v in g.nodes():
            f.write('v '+str(v)+' 2\n')
        #Iterate over all edges
        for edge in g.edges():
            f.write('e '+str(edge[0])+' '+str(edge[1])+' 2\n')
