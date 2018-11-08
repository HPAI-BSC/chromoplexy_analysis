'''
Generates graphs according to the window sizes for all files.
Decompose these into subgraphs, according to connected components, and stores in a format compatible for gSpan.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import printGraph, computeSubgraphs, generateNXGraph

import os
import networkx as nx

# Directory containing the files

data_path = '../data/allfiles'
output_path = '../data/allfiles_gspan'


def generate_all_patient_graphs(max_distance):
    subgraphs = []
    # Iterate over the files
    count = 0
    for filename in os.listdir(data_path):
        breaks, list_of_pairs = load_breaks(os.path.join(data_path, filename))
        if len(breaks) == 0:
            # print 'WARNING: Empty data file',file_name
            continue
        adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs,
                                                                                       max_distance)
        g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights, self_links=False,
                            connected_only=True)
        candidates = list(nx.connected_component_subgraphs(g))
        for c in candidates:
            if len(c.nodes()) >= 3 and len(c.nodes()) >= 3:
                print count, filename
                subgraphs.append(c)
                count += 1

    # Iterate over subgraphs and store
    with open(output_path + '/gspan_subgraphs_w' + str(max_distance) + '.txt', 'w') as f:
        counter = 0
        for g in subgraphs:
            f.write('t # ' + str(counter) + '\n')
            counter += 1
            # Iterate over vertices
            for v in g.nodes():
                f.write('v ' + str(v) + ' 2\n')
            # Iterate over all edges
            for edge in g.edges():
                f.write('e ' + str(edge[0]) + ' ' + str(edge[1]) + ' 2\n')


# subgraphs = []


for max_distance in [100, 1000, 10000, 100000, 1000000]:
    generate_all_patient_graphs(max_distance)
    # for file_name in os.listdir(data_path):
    #     breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
    #     if len(breaks) == 0:
    #         #print 'WARNING: Empty data file',file_name
    #         continue
    #     adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=w)
    #     subgraphs += computeSubgraphs(adjacency_matrix, vertex_labels)
    # #Iterate over subgraphs and store
    # with open('../data/exp3_subgraph_generator/all_subgraphs_'+str(w)+'.txt','w') as f:
    #     counter = 0
    #     for g in subgraphs:
    #         #skip two-vertices graph
    #         #if len(g.nodes()) < 3:
    #         #    continue
    #         f.write('t # '+str(counter)+'\n')
    #         counter+=1
    #         #Iterate over vertices
    #         for v in g.nodes():
    #             f.write('v '+str(v)+' 2\n')
    #         #Iterate over all edges
    #         for edge in g.edges():
    #             f.write('e '+str(edge[0])+' '+str(edge[1])+' 2\n')
