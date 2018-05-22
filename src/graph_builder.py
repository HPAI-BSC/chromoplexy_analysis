
def generateGraph(breaks, list_of_pairs, max_distance=100000):
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
    '''
    vertex_labels, vertex_ranges = generateVertices(breaks, max_distance)
    adjacency_matrix = generateEdges(list_of_pairs, vertex_labels, vertex_ranges)
    return adjacency_matrix, vertex_labels, vertex_ranges



def generateVertices(breaks, max_distance=100000):
    '''
    Input:
        breaks: dictionary {str:[int]}, where keys are chromosome ids
            and the corresponding list contains the position of breaks
            in that chromosome (sorted).
        max_distance: int, Breaks closer than max_distance are added to the same
            vertex.
    Output:
        vertex_labels: [str]. List of vertices generated. The value in position x 
            indicates the chromosome that vertex belongs to.
        vertex_ranges: [(int,int)]. List of vertices generated. The value in position 
            x indicates the range of values of that vertex. That is: 
                (first_break - max_distance, last_break + max_distance)
    '''
    #Create variables
    vertex_labels = []
    vertex_ranges = []
    #For each chromosome in the dictionary
    for k in breaks.keys():
        #If there is no breaks, continue
        if breaks[k] == []:
            continue
        #Keep current label
        current_label = k
        #Initialize the first vertex with the first break
        first_break = breaks[k][0]
        current_break = breaks[k][0]
        #Process the rest of breaks
        for b in breaks[k][1:]:
            #If the next break is within max_distance, add it to the vertex
            if b < current_break + max_distance:
                current_break = b
            #Otherwise, the vertex is ended. Store and initialize the next one
            else:
                #Store the vertex label and its range
                vertex_labels.append(k)
                #TODO: This may report ranges larger than the actual chromosome length. Add a min with a constant.
                vertex_ranges.append((max(first_break - max_distance,0), current_break + max_distance))
                #Initialize the next vertex
                first_break = b
                current_break = b
        #Store the last vertex
        vertex_labels.append(k)
        #TODO: This may report ranges larger than the actual chromosome length. Add a min with a constant.
        vertex_ranges.append((max(first_break - max_distance,0), current_break + max_distance))
    return vertex_labels, vertex_ranges


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
    adjacency_mat = np.zeros((len(vertex_ranges),len(vertex_ranges)))
    #For each pair within a break
    for p in list_of_pairs:
        #Get the pair of chromosome-position
        c1 = p[0][0]
        c1_pos = p[0][1]
        c2 = p[1][0]
        c2_pos = p[1][1]
        #Find vertex indices of first chromosome
        c1_all_indices = [idx for idx,x in enumerate(vertex_labels) if x == c1]
        #Get index of the vertex associated with the position
        vertex_c1_idx = -1
        for i in c1_all_indices:
            #If the vertex range fits the pos
            if vertex_ranges[i][0] < c1_pos and vertex_ranges[i][1] > c1_pos:
                #If a vertex was already found, something's wrong
                if vertex_c1_idx != -1:
                    raise Exception("Chromosome",c1,"and position",c1_pos,'belongs to more than one vertex:',i,'and',vertex_c1_idx)
                vertex_c1_idx = i
        #Same for the second chromosome
        c2_all_indices = [idx for idx,x in enumerate(vertex_labels) if x == c2]
        vertex_c2_idx = -1
        for i in c2_all_indices:
            if vertex_ranges[i][0] < c2_pos and vertex_ranges[i][1] > c2_pos:
                if vertex_c2_idx != -1:
                    raise Exception("Chromosome",c2,"and position",c2_pos,'belongs to more than one vertex:',i,'and',vertex_c2_idx)
                vertex_c2_idx = i
        adjacency_mat[vertex_c1_idx][vertex_c2_idx] += 1
        adjacency_mat[vertex_c2_idx][vertex_c1_idx] += 1
    return adjacency_mat 
        
