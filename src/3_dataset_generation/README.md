# STEP 2: Dataset Generation

This code generates a csv with the next dataset information:

**Goal Dataset:**
* chr_1, chr_2, ...,chr_X, chr_Y: The number of breaks of the chromosome on this patient. 
* DUP, DEL, TRA, h2hINV, t2tINV: The number of the breaks of this kind. 
* One feature for each of the possible translocation with the cuantity of this translocations on this patient. 
* DEL_1, ... DEL_X, DEL_Y: The number of deletions on the target chromosome. 
* DUP_1, ... DUP_X, DUP_Y: The number of duplications on the target chromosome. 
* Number of breaks: The total number of breaks of the patient. 
* Connected components: The number of connected components of the translocation graph.
* Connected components max size: The maximum size of the connected components of the translocation graph.
* The metadata features

**Files**
* dataset_generation.py
* dataset_generation.ipynb
* main_clique_analyisis.py
* main_freq_subgraph_plotter.py
* main_gspan_subgraph_genrator.py