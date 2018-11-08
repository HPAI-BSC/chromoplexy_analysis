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
        c1 = p[0][0]
        c1_pos = p[0][1]
        c2 = p[1][0]
        c2_pos = p[1][1]
        # Find vertex indices of first chromosome
        c1_all_indices = [idx for idx, x in enumerate(vertex_labels) if x == c1]
        # Get index of the vertex associated with the position
        vertex_c1_idx = -1
        for i in c1_all_indices:
            # If the vertex range fits the pos
            if vertex_ranges[i][0] < c1_pos and vertex_ranges[i][1] > c1_pos:
                # If a vertex was already found, something's wrong
                if vertex_c1_idx != -1:
                    raise Exception("Chromosome", c1, "and position", c1_pos, 'belongs to more than one vertex:', i,
                                    'and', vertex_c1_idx)
                vertex_c1_idx = i
        # Same for the second chromosome
        c2_all_indices = [idx for idx, x in enumerate(vertex_labels) if x == c2]
        vertex_c2_idx = -1
        for i in c2_all_indices:
            if vertex_ranges[i][0] < c2_pos and vertex_ranges[i][1] > c2_pos:
                if vertex_c2_idx != -1:
                    raise Exception("Chromosome", c2, "and position", c2_pos, 'belongs to more than one vertex:', i,
                                    'and', vertex_c2_idx)
                vertex_c2_idx = i
        adjacency_mat[vertex_c1_idx][vertex_c2_idx] += 1
        adjacency_mat[vertex_c2_idx][vertex_c1_idx] += 1
    return adjacency_mat
