'''
This code generates a graphs from the breaks data. Given a clique size, computes and stores the number of cliques, their chromosomic location, and their range within the chromosome, per patient.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import generateNXGraph
import networkx as nx

import os
import sys

if len(sys.argv)!=3:
    raise Exception('This function must be called with two parameters. An integer indicating the length of the sliding window, and an integer with the clique size.')

max_distance = int(sys.argv[1])
clique_size = int(sys.argv[2])

#Output files
file1 = open('../results/'+str(clique_size)+'-cliques_number_per_patient_w'+str(max_distance)+'.csv','w')
file2 = open('../results/'+str(clique_size)+'-cliques_location_per_patient_w'+str(max_distance)+'.csv','w')
file3 = open('../results/'+str(clique_size)+'-cliques_ranges_per_patient_w'+str(max_distance)+'.csv','w')

#Directory containing the files
data_path = '../data/allfiles'
#Iterate over the files
for filename in os.listdir(data_path):
    #Load the brakes
    breaks, list_of_pairs = load_breaks(os.path.join(data_path,filename))
    if len(breaks) == 0:
        print 'WARNING: Empty data file',filename
        continue
    #Generate the vertices and edges
    adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance)
    #Create the graph
    g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, self_links=False, connected_only=True)
    #List all cliques
    cliques = [x for x in list(nx.enumerate_all_cliques(g)) if len(x)==clique_size]
    file1.write(filename+','+str(len(cliques))+'\n')
    #List attributes of vertices in cliques
    vertex_chromosomes = nx.get_node_attributes(g,'chromosome')
    vertex_ranges = nx.get_node_attributes(g,'range')
    chromosome_list = []
    range_list = []
    for c in cliques:
        for v in c:
            chromosome_list.append(vertex_chromosomes[v])
            range_list.append(vertex_ranges[v])
    file2.write(filename+','+str(chromosome_list)+'\n')
    file3.write(filename+','+str(range_list)+'\n')

file1.close()
file2.close()
file3.close()

