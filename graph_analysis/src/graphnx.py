import networkx as nx


def generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights, self_links=False,
                    connected_only=True):
    '''
    Generates and returns a networkX graph with the input vertex labels and
    adjacency matrix. The graph is undirected.

    Input:
        adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices.
            A 0 indicates disconnection. Larger values indicate the number of breaks
            going from one vertex to the other.
            The matrix is symetrical, and the diagonal contains self-edges.
        vertex_labels: [str]. Of length equal to the length of adjacency matrix.
            The value in position x indicates the chromosome that the vertex at
            position x belongs to.
        vertex_ranges: [(int,int)]. List of vertices generated.
            The value in position x indicates the range of values of that vertex. That is:
                (first_break - max_distance, last_break + max_distance)
        vertex_weights: [(int)]	 List of the weight of each vertex. The value in position x indicates the weight of that
            vertex. This weight represents the number of breaks of this vertex.
        self_links: boolean. Are self links implemented in the graph? Defaults to False.
        connected_only: boolean. Remove all isolated vertices. Defaults to True.
    Output:
    '''
    import numpy as np
    if not self_links:
        # Remove self-edges
        np.fill_diagonal(adjacency_matrix, 0)
    # Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    # Create dictionaries of attributes
    dic_labels = {}
    dic_ranges = {}
    dic_weights = {}
    for n, v, r, w in zip(x.nodes(), vertex_labels, vertex_ranges, vertex_weights):
        dic_labels[n] = v
        dic_ranges[n] = r
        dic_weights[n] = w
    nx.set_node_attributes(x, dic_labels, 'chromosome')
    nx.set_node_attributes(x, dic_ranges, 'range')
    nx.set_node_attributes(x, dic_weights, 'weights')
    # Undirected, to avoid redundancy of symmetric matrix
    x.to_undirected()
    # Remove isolated vertices if requested
    if connected_only:
        x.remove_nodes_from(list(nx.isolates(x)))
    return x


def printGraph(adjacency_matrix, vertex_labels):
    '''
    Input
    adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices. 
        implies there is no edges between the corresponding vertices. Larger
        values indicate the number of breaks going from one vertex to the other.
        The matrix is symmetrical, and the diagonal may contain self-edges.
    ex_labels: [str]. Of length equal to the length of adjacency matrix. 
        The value in position x indicates the chromosome that the vertex at  
    position x belongs to.
    '''
    from matplotlib import pyplot as plt
    import networkx as nx
    import numpy as np
    # Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    # Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    # Create dictionary of labels to be added
    dic_labels = {}
    for n, v in zip(x.nodes(), vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, dic_labels, 'chromosome')
    # Undirected, to avoid duplicity
    x.to_undirected()
    # print 'Number of vertices in networkX', x.number_of_nodes()
    # print 'Number of edges in networkX', x.number_of_edges()
    # Plot the graph, after removing isolated vertices
    x.remove_nodes_from(nx.isolates(x))
    # print 'Number of connected vertices', x.number_of_nodes()
    # print 'Number of connected components', nx.number_connected_components(x)
    # print 'Number of 3-cliques', len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==3])
    # print 'Number of 4-cliques', len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==4])
    # print 'Number of 5-cliques', len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==5])
    # Plot with matplotlib
    plt.figure(figsize=(7, 7))
    edges, weights = zip(*nx.get_edge_attributes(x, 'weight').items())
    pos = nx.spring_layout(x, k=0.5, iterations=30)
    # pos = nx.spectral_layout(x)
    # Print graph nodes and edges
    nx.draw(x, pos, edgelist=edges, edge_color=weights, node_size=700, width=5.0, edge_cmap=plt.cm.winter)
    # Set chromosome as node label and print them
    labels = nx.get_node_attributes(x, 'chromosome')
    nx.draw_networkx_labels(x, pos, labels=labels, font_size=20)
    # Show the graph
    plt.show()


def getCliques(adjacency_matrix, vertex_labels):
    '''
    Input
    adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices. 
        implies there is no edges between the corresponding vertices. Larger
        values indicate the number of breaks going from one vertex to the other.
        The matrix is symetrical, and the diagonal may contain self-edges. 
    ex_labels: [str]. Of length equal to the length of adjacency matrix. 
        The value in position x indicates the chromosome that the vertex at  
    position x belongs to.
    '''
    import matplotlib
    from matplotlib import pyplot as plt
    import networkx as nx
    import numpy as np
    # Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    # Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    # Create dictionary of labels to be added
    dic_labels = {}
    for n, v in zip(x.nodes(), vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, dic_labels, 'chromosome')
    # Undirected, to avoid duplicity
    x.to_undirected()
    # remove isolated vertices
    x.remove_nodes_from(nx.isolates(x))
    return len([c for c in list(nx.enumerate_all_cliques(x)) if len(c) == 3]), len(
        [c for c in list(nx.enumerate_all_cliques(x)) if len(c) == 4])


def getAllCliques(adjacency_matrix, vertex_labels):
    '''
    Returns the number of cliques in the graph. Starts with size 3 and keeps increasing the size until no cliques are found. Return structure is a dictionary where the key is the clique size, and the value is the number of cliques found.
    Input
    adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices. 
        implies there is no edges between the corresponding vertices. Larger
        values indicate the number of breaks going from one vertex to the other.
        The matrix is symetrical, and the diagonal may contain self-edges. 
    ex_labels: [str]. Of length equal to the length of adjacency matrix. 
        The value in position x indicates the chromosome that the vertex at  
    position x belongs to.
    '''
    import networkx as nx
    import numpy as np
    # Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    # Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    # Create dictionary of labels to be added
    dic_labels = {}
    for n, v in zip(x.nodes(), vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, dic_labels, 'chromosome')
    # Undirected, to avoid duplicity
    x.to_undirected()
    # remove isolated vertices
    x.remove_nodes_from(nx.isolates(x))
    cliques = {}
    current_size = 3
    while True:
        current_cliques = len([c for c in list(nx.enumerate_all_cliques(x)) if len(c) == current_size])
        cliques[current_size] = current_cliques
        if current_cliques == 0:
            break
        current_size += 1
    return cliques


def computeSubgraphs(adjacency_matrix, vertex_labels):
    '''
    Input
    adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices. 
        implies there is no edges between the corresponding vertices. Larger
        values indicate the number of breaks going from one vertex to the other.
        The matrix is symetrical, and the diagonal may contain self-edges. 
    ex_labels: [str]. Of length equal to the length of adjacency matrix. 
        The value in position x indicates the chromosome that the vertex at  
    position x belongs to.
    '''
    import networkx as nx
    import numpy as np
    # Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    # Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    # Create dictionary of labels to be added
    dic_labels = {}
    for n, v in zip(x.nodes(), vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, dic_labels, 'chromosome')
    # Undirected, to avoid duplicity
    x.to_undirected()
    # print 'Number of vertices in networkX', x.number_of_nodes()
    # print 'Number of edges in networkX', x.number_of_edges()
    # Plot the graph, after removing isolated vertices
    x.remove_nodes_from(nx.isolates(x))

    return list(nx.connected_component_subgraphs(x))
