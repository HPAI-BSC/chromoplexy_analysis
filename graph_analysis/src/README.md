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
|-- src 
|   |-- notebooks
|   |   |-- 
|   |-- loader.py # loads the breask
|   |-- graphnx.py # Contains the graph creation and representation code
|   |-- exp0_graph_plotter.py # plots ALL the patient graphs
|   |-- exp3_subgraph_generator.py # Generates the subgraphs files in a gspan compatible format
|   |--
|-- documentation
└-- .gitignore
````