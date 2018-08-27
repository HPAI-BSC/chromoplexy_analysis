# STEP 2

This code generates a csv with the next dataset information:

**Goal Dataset:**

patient_id | subgraph0 | subgraph1 | ... | subgraphN | metadata | type of cancer (target)

To construct this I use the next classes:

**Subgraph_instance:**

	id: int

	description: string ('v 0 1 v 2 8 e 0 2 ')

	patients: array(strings)

	support: int (This variable is the global support of the graph, i.e. respect all the patients)

**Patient instance:**

	id: string

	graphs: dict(description(string):support(int)) (This attribute is the local support of the graph, i.e. respect this specific patient)

**Data:**

	all_subgraphs: array(Graph_instance)

	existing_subgraphs = key: description value: id

	patients = array(Patient_Instance)

General working flow: 

Generate all data -> order it by global support -> select a subset of this graphs -> generate dataset.

For every patient:
        1. Generate the patient graph.
        2. Generate all the subgraph patterns of this graph using gspan.
        3. Generate the sample (in terms of subgraphs) of this patient.
            (dict\[sub_graph\] = number of sub_graphs of this type)
        4. Add to the general list the new subgraphs.
        5. Generates a list of the most common patterns for all the patients by removing the less common
        6. Use this data to generate the final dataset and save it on a csv.

The gspan implementation is taken from:
[1] https://github.com/Jokeren/DataMining-gSpan

TODO:
- Decide what support to use, for now I'm ussing 0.8.
- Check and correct the documentation.