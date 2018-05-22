'''
Find the number of cliques of arbitrary size for every graph. Done for the different window sizes
'''


from loader import load_breaks
from graph_builder import generateGraph
from graphnx import getAllCliques
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter

data_path = '../vcfshorts/allfiles'
#Lets compute the distributions for each window size
for w in [100,1000,10000,100000,1000000]:
    print '-------------------'
    print '--WINDOW SIZE '+str(w)+'--'
    print '-------------------'
    all_cliques = []
    #Open all the files
    for file_name in os.listdir(data_path):
        #Load breaks, adjacency_matrix, etc.
        breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
        adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=w)
        #Compute the number of cliques
        if len(vertex_labels)!=0:
            all_cliques.append(getAllCliques(adjacency_matrix,vertex_labels))
        #No vertex, no cliques
        else:
            all_cliques.append({})
    #Get all keys
    all_keys = []
    for x in all_cliques:
        all_keys += x.keys()
    all_keys = sorted(set(all_keys),key=int)
    #For each clique size, aggregate and print 
    for k in all_keys:
        #Find the data
        k_cliques = []
        for sample in all_cliques:
            if k in sample.keys():
                k_cliques.append(sample[k])
            else:
                k_cliques.append(0)
        #Print
        print '    '+ str(k)+'-CLIQUES'
        for key,value in Counter(sorted(k_cliques,key=int)).iteritems():
            print str(value)+' samples had '+str(key)+' '+str(k)+'-cliques'
        print '-------------------'
        #plt.clf()
        #plt.figure(figsize = (15,10))
        #n, bins, patches = plt.hist(k_cliques, facecolor='green', alpha=0.75)
        #plt.xlabel('Number of '+str(k)+'-cliques')
        #plt.ylabel('Frequency')
        #plt.title('Distribution of cliques of size '+str(k)+' using window size '+str(w))
        #plt.grid(True)
        #plt.savefig(str(k)+'-clique_distribution_W'+str(w)+'.jpeg')
    print 
