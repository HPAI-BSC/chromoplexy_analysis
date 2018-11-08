from loader import load_breaks
from graph_builder import generateGraph
from graphnx import getCliques
from matplotlib import pyplot as plt
import numpy as np
import os

#Load the cancer types
cancer_types = {}
with open('../FINALstats.txt') as f:
    f.next()
    for l in f:
        if len(l.split('\t'))!=6:
            raise Exception("Wrong number of fields (i.e., not 5) in line",l)
        sampleId, num_breaks, chromoplexy_binary, chromoplexy_num, cancer_type, country = l.split('\t')
        if cancer_type not in cancer_types.keys():
            cancer_types[cancer_type] = []
        cancer_types[cancer_type].append(sampleId)


counter_all = 0
counter_all_empty = 0
cliques_all_3 = []
cliques_all_4 = []

plot_position_x = 0
plot_position_y = 0
fig, ax = plt.subplots(5,8)

for t in cancer_types.keys():
    #print 'Working on cancer type',t
    counter = 0
    #counter_empty = 0
    cliques_3 = []
    cliques_4 = []
    data_path = '../vcfshorts/allfiles'
    #data_path = '../vcfshorts/chromexsamples'
    #file_name = 'fca3f7d0-2231-661c-e040-11ac0c4832fd.vcf.tsv'
    #file_name = 'feccee20-a62d-4152-b832-b9fdaca87a61.vcf_chromex.tsv'
    
    for file_name in os.listdir(data_path):
        if file_name.split('.')[0] in cancer_types[t]:
            breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
            #if len(breaks) == 0:
            #    #print 'WARNING: Empty data file',file_name
            #    counter_empty +=1
            
            adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=100)
            counter += 1
            if len(vertex_labels)!=0:
                c3,c4 = getCliques(adjacency_matrix,vertex_labels)
                cliques_3.append(c3)
                cliques_4.append(c4)
            else:
                cliques_3.append(0)
                cliques_4.append(0)

    #print 'Cancer type',t, 'Mean/std of 3 cliques',np.asarray(cliques_3).mean(),np.asarray(cliques_3).std(),'Mean/std of 4 cliques',np.asarray(cliques_4).mean(),np.asarray(cliques_4).std()
    print t, counter, np.around(np.asarray(cliques_3).mean(),decimals=2),np.around(np.asarray(cliques_3).std(),decimals=2),np.around(np.asarray(cliques_4).mean(),decimals=2),np.around(np.asarray(cliques_4).std(),decimals=2)
    counter_all += counter
    #counter_all_empty += counter_empty
    cliques_all_3 += cliques_3
    cliques_all_4 += cliques_4
    print 'Cliques',cliques_3
    print 'Max cliques',max(cliques_3)
    #Plot distribution
    bins = max(cliques_3)+1
    print 'bins',bins
    frq, edges = np.histogram(cliques_3, bins)
    print 'edges',edges
    #fig, ax = plt.subplots(5,8)
    ax[plot_position_x, plot_position_y].plot()
    #ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
    plt.xticks(np.diff(edges)/2.0+edges[:-1],np.arange(bins))

plt.show()

print 'ALL', counter_all,np.around(np.asarray(cliques_all_3).mean(),decimals=2),np.around(np.asarray(cliques_all_3).std(),decimals=2),np.around(np.asarray(cliques_all_4).mean(),decimals=2),np.around(np.asarray(cliques_all_4).std(),decimals=2)
