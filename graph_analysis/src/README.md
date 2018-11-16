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
|   |-- Example of use.ipynb
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
It runs using the next global variables: 

````
PATIENTS_PATH = '../data/allfiles/'
GSPAN_COMPATIBLE_GRAPHS_PATH = '../data/allfiles_gspan/'
PLOT_PATH = '../data/plots/'
SUGRAPHS_PATH = '../data/subgraphs/'
````


This is the main graph patter file. It contains all the necessary functions to generate the patient graphs and run 
gspan over them. 

* *generate_all_patient_graphs(max_distance)*: generates all the patient graphs using the given distance and saves it 
in the GSPAN_COMPATIBLE_GRAPHS_PATH folder. 
* *call_gspan_from_library(file_path)*: It runs the installed gspan over the graphs on *file_path* and saves the subgraphs on 
SUGRAPHS_PATH.
* call_gspan_locally(filepath, outputpath, gspan_main_path, name): Same but from a downloaded gspan. 
* _read_graphs_from_file(file_path): helper function to read the graphs. 
* save_graphs_to_pdf(file_path): Plots the graphs of the file on *file_path* into a pdf.
* save_graphs_to_png(file_name): Same but into a png. 


The gspan implementation used on this project is [https://github.com/betterenvi/gSpan]