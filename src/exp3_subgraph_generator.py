'''
Generates graphs according to the window sizes for all files. Decompose these into subgraphs, according to connected components, and stores in a format compatible for gSpan.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import printGraph, computeSubgraphs

import os
import networkx as nx

data_path = '../vcfshorts/allfiles'
#data_path = '../vcfshorts/chromexsamples'
#file_name = 'fca3f7d0-2231-661c-e040-11ac0c4832fd.vcf.tsv'
#file_name = 'feccee20-a62d-4152-b832-b9fdaca87a61.vcf_chromex.tsv'

subgraphs = []
for w in [100,1000,10000,100000,1000000]:
    for file_name in os.listdir(data_path):
        breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
        if len(breaks) == 0:
            #print 'WARNING: Empty data file',file_name
            continue
        adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=w)
        subgraphs += computeSubgraphs(adjacency_matrix, vertex_labels)
    #Iterate over subgraphs and store
    with open('../exp3_subgraph_generator/all_subgraphs_'+str(w)+'.txt','w') as f:
        counter = 0
        for g in subgraphs:
            #skip two-vertices graph
            #if len(g.nodes()) < 3:
            #    continue
            f.write('t # '+str(counter)+'\n')
            counter+=1
            #Iterate over vertices
            for v in g.nodes():
                f.write('v '+str(v)+' 2\n')
            #Iterate over all edges
            for edge in g.edges():
                f.write('e '+str(edge[0])+' '+str(edge[1])+' 2\n')
