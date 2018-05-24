'''
Compute the number of cliques of size 3 and 4 for the data of the different cancer types. Then aggregate that information for all cancer types. Cliques are searched in a graph generated with variable window size, from 100 to 1M.
'''


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

#Lets compute the distributions for each window size
for w in [100,1000,10000,100000,1000000]:
    cliques_all_3 = []
    cliques_all_4 = []
    #Initialize a multiplot-plot
    plot_position_x = 0
    plot_position_y = 0
    fig, ax = plt.subplots(8,5, figsize = (70,50))
    #For each cancer type
    for t in cancer_types.keys():
        #print 'Working on cancer type',t
        counter = 0
        cliques_3 = []
        data_path = '../vcfshorts/allfiles'
        #Open all the files
        for file_name in os.listdir(data_path):
            #If the file belongs to that type of cancer
            if file_name.split('.')[0] in cancer_types[t]:
                #Load breaks, adjacency_matrix, etc.
                breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
                adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=w)
                #Compute the number of cliques
                if len(vertex_labels)!=0:
                    c3,c4 = getCliques(adjacency_matrix,vertex_labels)
                    cliques_3.append(c3)
                #If it has no breaks, skip the calculus
                else:
                    cliques_3.append(0)
        #The terrible histogram
        #We define as outlier everything beyond 30
        frq, edges = np.histogram(cliques_3, np.arange(32))
        #Aggregate outliers on last bin
        frq[-1]+=len([x for x in cliques_3 if x>30])
        #Define the bars and their distances
        ax[plot_position_x,plot_position_y].bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
        #Prepare the ticks
        x_ticks = np.diff(edges)/2.0+edges[:-1]
        #Split ticks between major and minor
        ax[plot_position_x,plot_position_y].set_xticks(x_ticks[::2])
        ax[plot_position_x,plot_position_y].set_xticks(x_ticks[1::2], minor=True)
        #For the major, add the label
        ax[plot_position_x,plot_position_y].set_xticklabels((list(map(str,np.arange(30)))+['>=30'])[::2], fontsize=18)
        #Add title to subplot of multiplot-plot
        ax[plot_position_x,plot_position_y].set_title(t, fontsize= 30)
        #Print the ticks, both major and minor, of different size
        ax[plot_position_x,plot_position_y].tick_params('x', length=20, width=2, which='major')
        ax[plot_position_x,plot_position_y].tick_params('x', length=10, width=1, which='minor')
        #Update the position of the subplot in the multiplot-plot
        plot_position_x+=1
        if plot_position_x == 8:
            plot_position_x = 0
            plot_position_y+=1
        #Aggregate all values for plotting the general results
        cliques_all_3 += cliques_3
    fig.tight_layout()
    plt.savefig('../exp1_cliques_distribution/distribution_cliques_3_W'+str(w)+'.jpeg')
    plt.clf()
    #Overall histogram 
    plt.clf()
    plt.figure(figsize = (15,10))
    frq, edges = np.histogram(cliques_all_3, np.arange(224))
    plt.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
    plt.savefig('../exp1_cliques_distribution/general_distribution_cliques_3_W'+str(w)+'.jpeg')
    plt.clf()

    #Now the same for cliques of 4. I know, I was just lazy...
    plot_position_x = 0
    plot_position_y = 0
    fig, ax = plt.subplots(8,5, figsize = (70,50))
    #For each cancer type
    for t in cancer_types.keys():
        #print 'Working on cancer type',t
        counter = 0
        cliques_3 = []
        cliques_4 = []
        data_path = '../vcfshorts/allfiles'
        #Open all the files
        for file_name in os.listdir(data_path):
            #If the file belongs to that type of cancer
            if file_name.split('.')[0] in cancer_types[t]:
                #Load breaks, adjacency_matrix, etc.
                breaks, list_of_pairs = load_breaks(os.path.join(data_path,file_name))
                adjacency_matrix, vertex_labels, vertex_ranges = generateGraph(breaks, list_of_pairs, max_distance=w)
                #Cmpute the number of cliques
                if len(vertex_labels)!=0:
                    c3,c4 = getCliques(adjacency_matrix,vertex_labels)
                    cliques_4.append(c4)
                #If it has no breaks, skip the calculus
                else:
                    cliques_4.append(0)
        #The terrible histogram
        #We define as outlier everything beyond 30
        frq, edges = np.histogram(cliques_4, np.arange(12))
        #Aggregate outliers on last bin
        frq[-1]+=len([x for x in cliques_4 if x>10])
        #Define the bars and their distances
        ax[plot_position_x,plot_position_y].bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
        #Prepare the ticks
        x_ticks = np.diff(edges)/2.0+edges[:-1]
        #Split ticks between major and minor
        ax[plot_position_x,plot_position_y].set_xticks(x_ticks[::2])
        ax[plot_position_x,plot_position_y].set_xticks(x_ticks[1::2], minor=True)
        #For the major, add the label
        ax[plot_position_x,plot_position_y].set_xticklabels((list(map(str,np.arange(10)))+['>=30'])[::2], fontsize=18)
        #Add title to subplot of multiplot-plot
        ax[plot_position_x,plot_position_y].set_title(t, fontsize= 30)
        #Print the ticks, both major and minor, of different size
        ax[plot_position_x,plot_position_y].tick_params('x', length=20, width=2, which='major')
        ax[plot_position_x,plot_position_y].tick_params('x', length=10, width=1, which='minor')
        #Update the position of the subplot in the multiplot-plot
        plot_position_x+=1
        if plot_position_x == 8:
            plot_position_x = 0
            plot_position_y+=1
        #Aggregate all values for plotting the general results
        cliques_all_4 += cliques_4
    fig.tight_layout()
    plt.savefig('../exp1_cliques_distribution/distribution_cliques_4_W'+str(w)+'.jpeg')
    plt.clf()
    #Overall histogram 
    plt.figure(figsize = (15,10))
    frq, edges = np.histogram(cliques_all_4, np.arange(224))
    plt.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
    plt.savefig('../exp1_cliques_distribution/general_distribution_cliques_4_W'+str(w)+'.jpeg')
    plt.clf()
