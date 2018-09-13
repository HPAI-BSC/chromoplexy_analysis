"""
The objective of this code is to analyze the traslocation graphs.

Process:
    - Generate the gspan compatible graphs 'v vid vertex_label' 'e vid0 vid1 edge_label'
    - Connected component histogram + mean size between components -> to features
    - Centrality study -> Ranking of the most central vertex -> Feature: central_rank_chrom_1 = [1,23] rank
    - Number of triangles per patient (gspan)
    - Gspan (less than 4 vertex)
"""

