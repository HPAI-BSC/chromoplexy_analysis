'''
Generate a graph from each patient data, and plot it.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import printGraph

import os
import sys

if len(sys.argv)!=2:
    raise Exception('This function must be called with one parameter. An integer indicating the length of the window.')

max_distance = int(sys.argv[1])



#Directory containing the files
data_path = '../data/allfiles'
#Iterate over the files
for file_name in os.listdir(data_path):
    #Load the brakes
    breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
    if len(breaks) == 0:
        print 'WARNING: Empty data file',file_name
        continue
    #Generate the graph
    adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance)
    #Print the graph
    printGraph(adjacency_matrix, vertex_labels)

