'''

'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import generateNXGraph, computeSubgraphs

import os
import sys

if len(sys.argv)!=3:
    raise Exception('This function must be called with two parameters. An integer indicating the length of the sliding window, and an integer with the clique size.')

max_distance = int(sys.argv[1])
clique_size = int(sys.argv[2])

different, same = 0, 0
#Directory containing the files
data_path = '../data/allfiles'
#Iterate over the files
for file_name in os.listdir(data_path):
    #Load the brakes
    breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
    if len(breaks) == 0:
        print 'WARNING: Empty data file',file_name
        continue
    #Generate the vertices and edges
    adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance)
    #Create the graph
    g = generateNXGraph(adjacency_matrix, vertex_labels, self_links=False, connected_only=True)

    #Count cliques per case
    print file_name
    print g.nodes()
    print len(computeSubgraphs(g))
print 'diff',different
print 'same',same
