define a generate graph method, which is common to all three methods.

def printGraph(adjacency_matrix, vertex_labels):
    '''
    Plots the inputed graph using networkX
    Input:
        adjancency_matrix: Squared 2D numpy matrix. Rows/Columns correspond to vertices. 
            A 0 indicates disconnection. Larger values indicate the number of breaks 
            going from one vertex to the other.
            The matrix is symetrical, and the diagonal contains self-edges. 
        vertex_labels: [str]. Of length equal to the length of adjacency matrix. 
            The value in position x indicates the chromosome that the vertex at  
            position x belongs to.
    '''
    import matplotlib
    from matplotlib import pyplot as plt
    import networkx as nx
    import numpy as np
    #Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    #Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    #Create dictionary of labels to be added
    dic_labels = {}
    for n,v in zip(x.nodes(),vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, dic_labels, 'chromosome')
    #Undirected, to avoid redundancy of symmetric matrix
    x.to_undirected()
    #print '|V| in networkX', x.number_of_nodes()
    #print '|E| in networkX', x.number_of_edges()
    #Plot the graph, after removing isolated vertices
    x.remove_nodes_from(nx.isolates(x))
    #print 'Number of connected vertices', x.number_of_nodes()
    #print 'Number of connected components', nx.number_connected_components(x)
    #print 'Number of 3-cliques', len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==3])
    #print 'Number of 4-cliques', len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==4])
    #print 'Number of 5-cliques', len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==5])
    #Plot with matplotlib 
    plt.figure(figsize=(7,7))
    edges,weights = zip(*nx.get_edge_attributes(x,'weight').items())
    pos = nx.spring_layout(x,k=0.5,iterations=30)
    #pos = nx.spectral_layout(x)
    #Print graph nodes and edges
    nx.draw(x, pos, edgelist=edges, edge_color=weights, node_size = 700, width=5.0, edge_cmap=plt.cm.winter)
    #Set chromosome as node label and print them
    labels = nx.get_node_attributes(x, 'chromosome')
    nx.draw_networkx_labels(x, pos, labels=labels, font_size = 20)
    #Show the graph
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
    #Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    #Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    #Create dictionary of labels to be added
    dic_labels = {}
    for n,v in zip(x.nodes(),vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, 'chromosome', dic_labels)
    #Undirected, to avoid duplicity
    x.to_undirected()
    #remove isolated vertices
    x.remove_nodes_from(nx.isolates(x))
    return len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==3]), len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==4])

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
    import matplotlib
    from matplotlib import pyplot as plt
    import networkx as nx
    import numpy as np
    #Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    #Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    #Create dictionary of labels to be added
    dic_labels = {}
    for n,v in zip(x.nodes(),vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, 'chromosome', dic_labels)
    #Undirected, to avoid duplicity
    x.to_undirected()
    #remove isolated vertices
    x.remove_nodes_from(nx.isolates(x))
    cliques = {}
    current_size = 3
    while True:
        current_cliques = len([c for c in list(nx.enumerate_all_cliques(x)) if len(c)==current_size])
        cliques[current_size] = current_cliques
        if current_cliques == 0:
            break
        current_size+=1
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
    import matplotlib
    from matplotlib import pyplot as plt
    import networkx as nx
    import numpy as np
    #Remove self-edges
    np.fill_diagonal(adjacency_matrix, 0)
    #Create graph
    x = nx.from_numpy_matrix(adjacency_matrix)
    #Create dictionary of labels to be added
    dic_labels = {}
    for n,v in zip(x.nodes(),vertex_labels):
        dic_labels[n] = v
    nx.set_node_attributes(x, 'chromosome', dic_labels)
    #Undirected, to avoid duplicity
    x.to_undirected()
    #print 'Number of vertices in networkX', x.number_of_nodes()
    #print 'Number of edges in networkX', x.number_of_edges()
    #Plot the graph, after removing isolated vertices
    x.remove_nodes_from(nx.isolates(x))

    return list(nx.connected_component_subgraphs(x))
