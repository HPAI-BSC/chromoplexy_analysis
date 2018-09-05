'''
Methods for generating graph representations (vertices and edges) from breaks
'''


def generateGraph(breaks, list_of_pairs, max_distance):
    '''
    Input:
        breaks: dictionary {str:[int]}, where keys are chromosome ids
            and the corresponding list contains the position of breaks
            in that chromosome (sorted).
        list_of_pairs: list[((str,int),(str,int))]. List of breaks, each entry contains
            first chromosome and position within, second chromosome and position within
        max_distance: int, Breaks closer than max_distance are added to the same
            vertex.
    Output:
        adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices.
            0 implies there is no edges between the corresponding vertices.
            Larger values indicate the number of breaks going from one vertex
            to the other. The matrix is symetrical, and the diagonal may contain
            self-edges.
        vertex_labels: [str]. Of length equal to the length of adjacency matrix.
            The value in position x indicates the chromosome that the vertex at
            position x belongs to.
        vertex_ranges: [(int,int)]. Of length equal to the length of adjacency matrix.
            The value in position x indicates the range of values of the vertex at
            position x. That is (first_break - max_distance, last_break + max_distance)
        vertex_weights: [int]. List of the weight of each vertex. The value in position x indicates the weight of that
            vertex. This weight represents the number of breaks of this vertex.
    '''
    vertex_labels, vertex_ranges, vertex_weights = generateVertices(breaks, max_distance)
    adjacency_matrix = generateEdges(list_of_pairs, vertex_labels, vertex_ranges)
    return adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights


def generateVertices(breaks, max_distance):
    '''
    Computes the vertices that exist within a list of chromosomic breaks. Each vertex contains a list of breaks.
    Once the first one is assigned, other may be added if these are within max_distance of the previous one.
    Input:
        breaks: dictionary {str:[int]}, where keys are chromosome ids
            and the corresponding non-empty list contains the position of breaks in that chromosome (sorted).
        max_distance: int, Breaks closer than max_distance are added to the same vertex.
    Output:
        vertex_labels: [str]. List of vertices generated.
            The value in position x indicates the chromosome that vertex belongs to.
        vertex_ranges: [(int,int)]. List of vertices generated.
            The value in position x indicates the range of values of that vertex. That is:
                (first_break - max_distance, last_break + max_distance)
        vertex_weights: [int]. List of the weight of each vertex. The value in position x indicates the weight of that
            vertex. This weight represents the number of breaks of this vertex.
    '''
    # Create variables
    vertex_labels, vertex_ranges, vertex_weights = list(), list(), list()
    # For each chromosome in the dictionary
    for chromosome in breaks.keys():
        # Keep a list of breaks added to vertices
        added_breaks = []
        number_of_breaks = 0
        # Iterate until all breaks have been assigned to a vertex.
        while (len(added_breaks) != len(set(breaks[chromosome]))):
            # Find the first unassigned break
            vertex_seed = [x for x in breaks[chromosome] if x not in added_breaks][0]
            # Initialize the start/end of vertex counter around it
            ini_of_vertex = vertex_seed - max_distance
            end_of_vertex = vertex_seed + max_distance
            # Iterate until no more breaks can be added (at least one will)
            vertex_done = False
            while not vertex_done:
                vertex_done = True
                # Add all breaks within range
                for current_break in breaks[chromosome]:
                    # Skip the added ones
                    if current_break in added_breaks:
                        # print 'Skipping it, already added'
                        continue
                    # Check if its within range
                    if ini_of_vertex < current_break < end_of_vertex:
                        number_of_breaks += 1
                        added_breaks.append(current_break)
                        ini_of_vertex = min(ini_of_vertex, current_break - max_distance)
                        end_of_vertex = max(end_of_vertex, current_break + max_distance)
                        vertex_done = False
            # Once the vertex is done, store it
            vertex_labels.append(chromosome)
            # TODO: This may cause ranges larger than the full chromosome length. Add "min(end_of_vertex,X)))" with X provided by LS.
            vertex_ranges.append((max(ini_of_vertex, 0), end_of_vertex))
            vertex_weights.append(number_of_breaks)
    return vertex_labels, vertex_ranges, vertex_weights


def generateEdges(list_of_pairs, vertex_labels, vertex_ranges):
    '''
    Computes the adjacency matrix of the graph defined by the inputs.
    Its a squared, symmetric 2D matrix where 0 indicates disconnection.
    Input:
        list_of_pairs: list[((str,int),(str,int))]. List of breaks, each entry contains
            first chromosome and position within, second chromosome and position within
        vertex_labels: [str]. Of length equal to the length of adjacency matrix.
            The value in position x indicates the chromosome that the vertex at
            position x belongs to.
        vertex_ranges: [(int,int)]. Of length equal to the length of adjacency matrix.
            The value in position x indicates the range of values of the vertex at
            position x. That is (first_break - max_distance, last_break + max_distance)
    Output:
        adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices.
            0 implies there is no edges between the corresponding vertices.
            Larger values indicate the number of breaks going from one vertex
            to the other. The matrix is symetrical, and the diagonal may contain
            self-edges.
    '''
    import numpy as np
    adjacency_mat = np.zeros((len(vertex_ranges), len(vertex_ranges)))
    # For each pair within a break
    for p in list_of_pairs:
        # Get the pair of chromosome-position
        c1, c2 = p[0][0], p[1][0]
        c1_pos, c2_pos = p[0][1], p[1][1]
        # Find vertex indices of first chromosome
        c1_all_indices = [idx for idx, x in enumerate(vertex_labels) if x == c1]
        # Find the index of the vertex associated with the position
        # TODO: Change this search loop for a more efficient solution
        vertex_c1_idx = -1
        for i in c1_all_indices:
            # If the vertex range fits the pos
            if vertex_ranges[i][0] < c1_pos and vertex_ranges[i][1] > c1_pos:
                # If a vertex was already found, something's wrong
                if vertex_c1_idx != -1:
                    raise Exception("Chromosome", c1, "and position", c1_pos, 'belongs to more than one vertex:', i,
                                    'and', vertex_c1_idx)
                else:
                    vertex_c1_idx = i
                # Same for the second chromosome
        c2_all_indices = [idx for idx, x in enumerate(vertex_labels) if x == c2]
        vertex_c2_idx = -1
        for i in c2_all_indices:
            if vertex_ranges[i][0] < c2_pos and vertex_ranges[i][1] > c2_pos:
                if vertex_c2_idx != -1:
                    raise Exception("Chromosome", c2, "and position", c2_pos, 'belongs to more than one vertex:', i,
                                    'and', vertex_c2_idx)
                else:
                    vertex_c2_idx = i
        adjacency_mat[vertex_c1_idx][vertex_c2_idx] += 1
        adjacency_mat[vertex_c2_idx][vertex_c1_idx] += 1
    return adjacency_mat


def generateTRAGraph(patient, data_path, output_path='', connected_only=True, plot_graph=False):
    '''
    This function generates a graph per patient representing the traslocations of this patient.
    vertex: Chromosome
    edge: the number of traslocations between each chromosome

    Input:
        patient(string):  The patient file name.
        data_path(string): The path were the patient files are.
        output_path (string): The output path.
        connected_only (Bool): If true, removes the isolated nodes from the graph.
        plot_graph (Bool): If true, prints the graphs on the output path.
    Output:
        graph: networkx format
        adjacency_matrix: Pandas dataframe
    '''
    import pandas as pd
    from natsort import natsorted
    import numpy as np
    import networkx as nx
    from matplotlib import pyplot as plt
    import gc

    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', '20', '21', '22', 'X', 'Y']
    patient_path = data_path + patient
    # Load the patient breaks, and select only the traslocations
    patient_breaks = pd.read_csv(patient_path, sep='\t', index_col=None)
    patient_breaks['chrom2'] = patient_breaks['chrom2'].map(str)

    only_TRA = patient_breaks.loc[patient_breaks['svclass'] == 'TRA']

    # The crosstab is equivalent to the adjacency matrix, so we use this to calculate it
    ct_tra = pd.crosstab(only_TRA['#chrom1'], only_TRA['chrom2'])

    ct_tra.index = ct_tra.index.map(str)

    aux = pd.DataFrame(0,columns=chromosomes, index=chromosomes)
    aux.index = aux.index.map(str)

    ct_tra = aux.add(ct_tra,fill_value=0)
    aux = None
    # Reorder
    ct_tra = ct_tra.reindex(index=natsorted(ct_tra.index))
    ct_tra = ct_tra[chromosomes]
    # change the values to int
    ct_tra = ct_tra.astype(int)

    # Generate the adjacency matrix
    adjacency_matrix = pd.DataFrame(data=np.maximum(ct_tra.values, ct_tra.values.transpose()),
                                columns=chromosomes, index=chromosomes)
    # print(adjacency_matrix)
    graph = nx.from_pandas_adjacency(adjacency_matrix)
    graph.to_undirected()
    # Remove isolated vertices if requested
    if connected_only:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    if plot_graph:
        pos = nx.spring_layout(graph)
        print(nx.info(graph))
        # version 2
        plt.figure(figsize=(20,20))
        nx.draw(graph, pos, with_labels=True)
        # specifiy edge labels explicitly
        edge_labels = dict([((u, v,), d['weight'])
                            for u, v, d in graph.edges(data=True)])
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        # show graphs
        plt.savefig(output_path + patient.replace('.vcf.tsv','.png') )
        patient_breaks = None
        graph = None
        plt.close()
        plt.clf()
        gc.collect()
    return graph, adjacency_matrix



def test():
    import os
    data_path = '../../data/raw_original_data/allfiles/'
    # patient = '0b6cd7df-6970-4d60-b7b5-85002a7d8781.vcf.tsv'
    output_path = '../../data_chromosome/graphs/'
    for patient in os.listdir(data_path):
        print(patient)
        generateTRAGraph(patient,data_path, output_path,plot_graph=True)

