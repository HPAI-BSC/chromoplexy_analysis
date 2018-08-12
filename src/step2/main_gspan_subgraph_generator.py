'''
Reads chromoplexy breaks, generates a graph using a sliding window of fixed size, and stores all the resulting connected subraphs in a file compatible with an implementation of gSpan [1].
Discards all subgraphs with less than 3 vertices or 3 edges.
[1] https://github.com/betterenvi/gSpan
'''

import os
import sys

sys.path.insert(1, '../src')

from step1.loader import load_breaks
from step1.graph_builder import generateGraph
from step1.graphnx import generateNXGraph

import networkx as nx
from datetime import timedelta
import time

DATAPATH = '../../data'

try:
    os.mkdir(DATAPATH + '/results')
except:
    pass


def generate_all_patient_graphs(max_distance):
    subgraphs = []
    # Directory containing the files
    data_path = DATAPATH + '/allfiles'
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
    with open(DATAPATH + '/results/gspan_subgraphs_w' + str(max_distance) + '.txt', 'w') as f:
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


def generate_one_patient_graph(filename, max_distance,gspan_path='/allfiles_gspan_format/', with_vertex_weight=False, with_vertex_chromosome=False,
                               with_edge_weight=False):
    subgraphs = []
    # Directory containing the files
    data_path = DATAPATH + '/allfiles'
    # Iterate over the files
    count = 0

    breaks, list_of_pairs = load_breaks(os.path.join(data_path, filename))
    if len(breaks) == 0:
        # print 'WARNING: Empty data file',file_name
        return
    adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs,
                                                                                   max_distance)
    g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights, self_links=False,
                        connected_only=True)
    candidates = list(nx.connected_component_subgraphs(g))
    for c in candidates:
        if len(c.nodes()) >= 3 and len(c.nodes()) >= 3:
            subgraphs.append(c)
            count += 1

    # store
    try:
        os.mkdir(DATAPATH + gspan_path)
    except:
        pass

    with open(DATAPATH + gspan_path + filename + '.txt', 'w') as f:
        counter = 0
        for g in subgraphs:
            f.write('t # ' + str(counter) + '\n')
            counter += 1
            # Iterate over vertices
            for v in g.nodes():
                if with_vertex_weight:
                    weight = g.nodes[v]['weights']
                elif with_vertex_chromosome:
                    weight = g.nodes[v]['chromosome']
                else:
                    weight = 2
                f.write('v ' + str(v) + ' ' + str(weight) + '\n')
            # Iterate over all edges
            for edge in g.edges():
                if with_edge_weight:
                    weight = g.edges[edge]['weight']
                else:
                    weight = 2
                f.write('e ' + str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(weight) + '\n')


def main():
    if len(sys.argv) != 2:
        raise Exception(
            'This function must be called with one parameter. An integer indicating the length of the sliding window.')
    max_distance = int(sys.argv[1])
    generate_all_patient_graphs(max_distance)


if __name__ == '__main__':
    init = time.time()
    main()
    print('time:', timedelta(seconds=time.time() - init))
