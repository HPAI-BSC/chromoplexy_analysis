'''
Generate a graph from each patient data, and plot it.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import printGraph

import os

data_path = '../vcfshorts/allfiles'
#data_path = '../vcfshorts/chromexsamples'
file_name = 'fca3f7d0-2231-661c-e040-11ac0c4832fd.vcf.tsv'
file_name = 'feccee20-a62d-4152-b832-b9fdaca87a61.vcf_chromex.tsv'

for file_name in os.listdir(data_path):

    breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
    if len(breaks) == 0:
        print 'WARNING: Empty data file',file_name
        continue
    
    adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=100)
    
    printGraph(adjacency_matrix, vertex_labels)

