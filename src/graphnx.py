'''
This file contains all methods related with NetworkX functionalities
'''

import networkx as nx

def generateNXGraph(adjacency_matrix, vertex_labels, self_links=False, connected_only=True):
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
        self_links: boolean. Are self links implemented in the graph? Defaults to False.
        connected_only: boolean. Remove all isolated vertices. Defaults to True.
    Output:
    '''
    import numpy as np
    if self_links == False:
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
    #Remove isolated vertices if requested
    if connected_only:
        x.remove_nodes_from(nx.isolates(x))
    return x


def printGraph(graph):
    '''
    Plots the inputed graph using networkX
    Input:
        graph: networkx graph to plot
    '''
    import matplotlib
    from matplotlib import pyplot as plt
    #Plot with matplotlib 
    plt.figure(figsize=(7,7))
    edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
    pos = nx.spring_layout(graph,k=0.5,iterations=30)
    #pos = nx.spectral_layout(graph)
    #Print graph nodes and edges
    nx.draw(graph, pos, edgelist=edges, edge_color=weights, node_size = 700, width=5.0, edge_cmap=plt.cm.winter)
    #Set chromosome as node label and print them
    labels = nx.get_node_attributes(graph, 'chromosome')
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size = 20)
    #Show the graph
    plt.show()
    return


#def getCliques(graph):
#    '''
#    Input:
#        graph: networkx graph to compute cliques on
#    '''
#    import matplotlib
#    from matplotlib import pyplot as plt
#    return len([c for c in list(nx.enumerate_all_cliques(graph)) if len(c)==3]), len([c for c in list(nx.enumerate_all_cliques(graph)) if len(c)==4])
#
#
#def getAllCliques(graph):
#    '''
#    Returns the number of cliques in the graph. Starts with size 3 and keeps increasing the size until no cliques are found. Return structure is a dictionary where the key is the clique size, and the value is the number of cliques found.
#    Input:
#        graph: networkX graph to compute cliques on
#    '''
#    import matplotlib
#    from matplotlib import pyplot as plt
#    cliques = {}
#    current_size = 3
#    while True:
#        current_cliques = len([c for c in list(nx.enumerate_all_cliques(graph)) if len(c)==current_size])
#        cliques[current_size] = current_cliques
#        if current_cliques == 0:
#            break
#        current_size+=1
#    return cliques 
#
#
#def computeSubgraphs(graph):
#    '''
#    Input:
#        graph: networkX graph to compute subgraphs on
#    '''
#    import matplotlib
#    from matplotlib import pyplot as plt
#    return list(nx.connected_component_subgraphs(graph))


#UNTESTED METHOD
#def graphStatistics(graph):
#    '''
#    Print statistics of a graph.
#    Input:
#        graph: networkX graph to compute statistics on
#    '''
#    print '|V| in networkX', graph.number_of_nodes()
#    print '|E| in networkX', graph.number_of_edges()
#    print 'Number of connected vertices', graph.number_of_nodes()
#    print 'Number of connected components', nx.number_connected_components(graph)
#    print 'Number of 3-cliques', len([c for c in list(nx.enumerate_all_cliques(graph)) if len(c)==3])
#    print 'Number of 4-cliques', len([c for c in list(nx.enumerate_all_cliques(graph)) if len(c)==4])
#    print 'Number of 5-cliques', len([c for c in list(nx.enumerate_all_cliques(graph)) if len(c)==5])
#    return

