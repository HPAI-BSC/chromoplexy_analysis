'''
Generate a graph from each patient data, and plot it.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import printGraph

import os

data_path = '../data/allfiles/'

for file_name in os.listdir(data_path):
    breaks, list_of_pairs = load_breaks(os.path.join(data_path, file_name))
    if len(breaks) == 0:
        print 'WARNING: Empty data file', file_name
        continue

    adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=100)

    try:
        printGraph(adjacency_matrix, vertex_labels)
    except:
        pass
