'''
Generate a graph from each patient data, and plot it.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import generateNXGraph, printGraph
import numpy as np
import shutil
import os
import re

# import sys
# if len(sys.argv) != 2:
# 	raise Exception(
# 		'This function must be called with one parameter. An integer indicating the length of the sliding window.')
# max_distance = int(sys.argv[1])


# Directory containing the files
data_path = '../../data/allfiles'


def plot_all(max_distance):
    # Iterate over the files
    for file_name in os.listdir(data_path):
        # Load the brakes
        breaks, list_of_pairs = load_breaks(os.path.join(data_path, file_name))
        if len(breaks) == 0:
            print 'WARNING: Empty data file', file_name
            continue
        # Generate the vertices and edges
        adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs,
                                                                                       max_distance)
        # Create the graph
        g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights, self_links=False,
                            connected_only=True)
        # Print the graph
        print 'Showing graph of ', file_name
        printGraph(g, show_vertex_weights=False)

        print vertex_ranges


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def plot_vertex_weight_histogram(max_distance):
    """
    This function plots one histogram for the vertex weights and another for the chromosomes with the breaks
    :param max_distance:
    :return:
    """
    from matplotlib import pyplot as plt

    # Iterate over the files
    all_weights = np.array([], dtype=np.float)
    all_labels = np.array([])
    for file_name in os.listdir(data_path):
        # Load the brakes
        breaks, list_of_pairs = load_breaks(os.path.join(data_path, file_name))
        if len(breaks) == 0:
            print 'WARNING: Empty data file', file_name
            shutil.move(os.path.join(data_path, file_name), os.path.join('../../data/empty_files', file_name))
            continue
        # Generate the vertices and edges
        adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs,
                                                                                       max_distance)
        all_weights = np.append(all_weights, np.array(vertex_weights))
        all_labels = np.append(all_labels, np.array(vertex_labels))

    from collections import Counter

    all_w_dict = Counter(all_weights.astype(np.float))
    little = {k: all_w_dict[k] for k in all_w_dict.keys() if all_w_dict[k] >= 100}

    print(all_w_dict)

    all_lab_dict = sorted(Counter(all_labels).items(), key=lambda (k, v): (natural_key(k), v))
    x, y = zip(*all_lab_dict)

    # print np.array(all_w_dict.values()).astype(np.float)/np.max(all_w_dict.values())

    plt.figure()
    plt.bar(list(all_w_dict.keys()), np.array(all_w_dict.values()).astype(np.float) / np.max(all_w_dict.values()),
            color='b')
    plt.title('Weights ' + str(max_distance))
    plt.savefig('../../data/plots/test_weights' + str(max_distance) + '.png')
    plt.figure()
    plt.bar(list(little.keys()), np.array(little.values()).astype(np.float) / np.max(little.values()),
            color='b')
    plt.title('Weights little ' + str(max_distance))
    plt.savefig('../../data/plots/test_weights_little' + str(max_distance) + '.png')
    plt.figure()
    plt.bar(x, y, color='g')
    plt.title('Labels ' + str(max_distance))
    plt.savefig('../../data/plots/test_labels' + str(max_distance) + '.png')
    # plt.show()
    plt.close()


def plot_one_file(file_name, max_distance=1000):
    # Load the brakes
    breaks, list_of_pairs = load_breaks(os.path.join(data_path, file_name))
    if len(breaks) == 0:
        print 'WARNING: Empty data file', file_name

    # Generate the vertices and edges
    adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs, max_distance)
    # Create the graph
    g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights, self_links=False,
                        connected_only=True)
    # Print the graph
    print 'Showing graph of ', file_name
    printGraph(g, name=file_name, visualize=False, show_vertex_weights=False)


# print vertex_ranges

def main():
    test_file = 'e84e0649-a2e8-4873-9cb6-1aa65601ae3a.vcf.tsv'
    # plot_one_file(test_file,max_distance=1000)
    all_distances = [500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]
    for distance in [500, 10000]:
        plot_vertex_weight_histogram(distance)


if __name__ == '__main__':
    import time
    from datetime import timedelta

    init = time.time()
    main()
    print 'time:', timedelta(seconds=time.time() - init)
