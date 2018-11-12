# Graph Analysis
 
On this project we have all the necesary code to extract different paterns on a set of patient data. 
 
The code is structured the next way: 

````
graph_analysis
|-- data
│   |-- allfiles # The folder of the breaks
|   |   |-- 0a6be23a-d5a0-4e95-ada2-a61b2b5d9485.vcf.tsv
|   |   |-- ...
|   |-- allfiles_gspan
|   |-- subgraphs
|   └-- plots
|-- src 
|   |-- loader.py # loads the breask
|   |-- graph_builder.py 
|   |-- graphnx.py # Contains the graph creation and representation code
|   └-- graph_pattern_extraction.py
|-- documentation
└-- .gitignore
````

## loader.py

   * load_breaks(file_location): return breaks_by_chromosome, list_of_pairs

This file contains the loader function. It loads the breaks of the patients and returns a list of breaks and a list of 
patients. 

## graph_builder.py
This file contains the graph basic functions. 
   * generateGraph(breaks,list_of_pairs,max_distance): return adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights
   * generateVertices(breaks, max_distance): return vertex_labels, vertex_ranges, vertex_weights
   * generateEdges(list_of_pairs, vertex_labels, vertex_ranges): return adjacency_mat
   
## graphnx.py
This file contains the graphnx related functions. 

## graph_pattern_extraction.py