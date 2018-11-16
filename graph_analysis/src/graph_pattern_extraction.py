import os
import sys

import networkx as nx
from cStringIO import StringIO
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import generateNXGraph

PATIENTS_PATH = '../data/allfiles/'
GSPAN_COMPATIBLE_GRAPHS_PATH = '../data/allfiles_gspan/'
PLOT_PATH = '../data/plots/'
SUGRAPHS_PATH = '../data/subgraphs/'


def generate_all_patient_graphs(max_distance):
    """
    Generates graphs according to the window sizes for all files.
    Decompose these into subgraphs, according to connected components, and stores in a format compatible for gSpan.
    """
    subgraphs = []
    # Iterate over the files
    count = 0
    for filename in os.listdir(PATIENTS_PATH):
        breaks, list_of_pairs = load_breaks(os.path.join(PATIENTS_PATH, filename))
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
    with open(GSPAN_COMPATIBLE_GRAPHS_PATH + 'gspan_subgraphs_w' + str(max_distance) + '.txt', 'w') as f:
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


def call_gspan_from_library(file_path):
    """ Runs the gSpan algorithm to find frequent subgraphs """
    from gspan_mining.config import parser
    from gspan_mining.gspan import gSpan

    # args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' -u 4 -v False ' + '-p ' + str(plot) + ' ' + file_path
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    args_str = file_path + ' -l 3 ' + '-s 5000'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = gSpan(
        database_file_name=FLAGS.database_file_name,
        min_support=FLAGS.min_support,
        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
        max_num_vertices=FLAGS.upper_bound_of_num_vertices,
        max_ngraphs=FLAGS.num_graphs,
        is_undirected=(not FLAGS.directed),
        verbose=FLAGS.verbose,
        visualize=FLAGS.plot,
        where=FLAGS.where
    )

    gs.run()
    name = file_path.split('/')[-1]
    with open(SUGRAPHS_PATH + name.replace('.txt', '_frequent_subgraphs.txt'), 'w') as f:
        f.write(redirected_output.getvalue())
    sys.stdout = old_stdout


def call_gspan_locally(filepath, outputpath, gspan_main_path, name):
    """ Runs the gSpan algorithm to find frequent subgraphs """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    sys.argv = [gspan_main_path, filepath, '-l 3', '-s 5000']
    try:
        execfile(gspan_main_path)
    except:
        raise
    with open(outputpath + name, 'w') as f:
        f.write(redirected_output.getvalue())
    f.close()
    sys.stdout = old_stdout


def _generateNXGraphLite(adj_mat):
    x = nx.from_numpy_matrix(adj_mat)
    return x


def _read_graphs_from_file(file_path):
    """ Reads the most common subgraphs and turns them into a nx graphs"""
    with open(file_path) as f:
        graphs = []
        supports = []
        for l in f:
            # Skip empty lines
            if l == '\n':
                continue
            # If its a new graph. Store the current one and initialize the next
            if l[0] == '-':
                graphs.append(_generateNXGraphLite(adj_mat))
                supports.append(support)
            if l[0] == 't':
                # Initialize next graph
                support = 0
                first_edge = True
                vertices = []
            # Its a new vertex. Add it.
            if l[0] == 'v':
                vertices.append((int(l.split(' ')[1])))
            # Its a new edge. Add it.
            if l[0] == 'e':
                if first_edge:
                    adj_mat = np.zeros((len(vertices), len(vertices)))
                    first_edge = False
                # Add edge
                # print(l, adj_mat)
                adj_mat[int(l.split(' ')[1]), int(l.split(' ')[2])] += 1
            # Its the graph support. Store it.
            if l[0] == 'S':
                support = int(l.split(' ')[1])

    # Order graphs by support
    graphs = [x for _, x in sorted(zip(supports, graphs), reverse=True)]
    supports = sorted(supports, reverse=True)
    return graphs, supports


def save_graphs_to_pdf(file_path):
    """ Plots the common subgraphs to a pdf"""
    graphs, supports = _read_graphs_from_file(file_path)
    name = file_path.split('/')[-1]
    with PdfPages(PLOT_PATH + name.replace('.txt', '.pdf')) as pdf:
        for graph, support in zip(graphs, supports):
            plt.figure(figsize=(7, 7))
            plt.axis('off')
            edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
            pos = nx.spring_layout(graph, k=0.5, iterations=30)
            nx.draw(graph, pos, edgelist=edges, edge_color=weights,
                    node_size=700, width=5.0, edge_cmap=plt.cm.winter)
            # nx.draw_networkx_labels(graph, pos, labels=labels, font_size = 20)
            plt.title('Support: ' + str(support))
            pdf.savefig()
            plt.close()


def save_graphs_to_png(file_name):
    """ Plots the common subgraphs to a png"""
    graphs, supports = _read_graphs_from_file(file_name)
    name = file_name.split('/')[-1]
    plt_num = 1
    y_size = len(graphs) * 6
    fig = plt.figure(figsize=(15, y_size))  # inches
    for graph, support in zip(graphs, supports):
        plt.subplot(len(graphs) / 2, 2, plt_num)
        plt.axis('off')
        edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
        pos = nx.spring_layout(graph, k=0.5, iterations=30)
        nx.draw(graph, pos, edgelist=edges, edge_color=weights,
                node_size=700, width=5.0, edge_cmap=plt.cm.winter)
        # nx.draw_networkx_labels(graph, pos, labels=labels, font_size = 20)
        plt.title('Support: ' + str(support))
        plt_num += 1
    plt.savefig(PLOT_PATH + name.replace('.txt', '.png'))


def main():
    distances = [100, 1000]
    # First we generate the subgraph files fore every distance.
    #for max_distance in distances:
    #    generate_all_patient_graphs(max_distance)

    # Extract the common subgraphs of the wanted distance using gspan and save it to a file.
    for d in distances:
        f = '../data/allfiles_gspan/gspan_subgraphs_w' + str(d) + '.txt'
        call_gspan_from_library(f)

    # Plot the common subgraphs
    for d in distances:
        f = '../data/subgraphs/gspan_subgraphs_w' + str(d) + '_frequent_subgraphs.txt'
        save_graphs_to_pdf(f)
        save_graphs_to_png(f)


if __name__ == "__main__":
    main()
