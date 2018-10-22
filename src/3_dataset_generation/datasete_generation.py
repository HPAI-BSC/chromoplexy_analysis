"""
This code generates a csv with the next dataset information:
Goal Dataset:
patient_id | subgraph0 | subgraph1 | ... | subgraphN | metadata | type of cancer (target)

To construct this I use the next classes:

Subgraph_instance:
    id: int
    description: string ('v 0 1 v 2 8 e 0 2 ')
    patients: array(strings)
    support: int (This variable is the global support of the graph, i.e. respect all the patients)

Patient instance:


Data:
    array(Graph)

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
    - Decide what support to use, for now I'm using 0.8
"""

import os
import sys

sys.path.insert(1, '../../src')

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from step1.graph_builder import generateTRAGraph

from datetime import timedelta
import time
import numpy as np
import pandas as pd
import networkx as nx


def generate_dataset(patients, data_path, output_path, name='classification_dataset.csv'):
    """
        This function generates a dataset from the patient names on "patients" and saves it in "output path"
         as a csv with the name "name".

        This dataset has the next features
            * chr_1, chr_2, ...,chr_X, chr_Y: The number of breacks of the chromosome on this patient.
            * DUP, DEL, TRA, h2hINV, t2tINV: The number of the breaks of this kind.
            * One feature for each of the possible translocation with the cuantity of this translocations on this patient.
            * DEL_1, ... DEL_X, DEL_Y: The number of deletions on the target chromosome.
            * DUP_1, ... DUP_X, DUP_Y: The number of duplications on the target chromosome.
            * Number of breaks: The total number of breaks of the patient.
            * Connected components: The number of connected components of the translocation graph.
            * Connected components max size: The maximum size of the connected components of the translocation graph.
            * The metadata features
    """

    print('Generating csv..')
    metadata = pd.read_csv(data_path + '/raw_original_data/metadatos_v2.0.csv')
    metadata = metadata.set_index('sampleID')
    # Remove the patients that doesn't have metadata
    l = len(patients)
    patients = [p for p in patients if p in list(metadata.index)]
    print('There are ', l-len(patients) , 'patients that do not appear in metadata')
    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', '20', '21', '22', 'X', 'Y']
    svclass = ['DUP', 'DEL', 'TRA', 'h2hINV', 't2tINV']
    graph_columns = ['(' + chromosomes[i] + ',' + chromosomes[j] + ')' for i in range(len(chromosomes))
                                     for j in range(len(chromosomes)) if i < j]

    all_columns = np.append(metadata.columns, graph_columns).flatten()

    # initialize the dataset and append the metadata to it
    dataset = pd.DataFrame(columns=all_columns)
    dataset = pd.concat([dataset, metadata])
    # initialize the graph related columns
    # dataset.loc[:, graph_columns] = 0

    i = 0
    for patient in metadata.index:
        g, edge_list, matrix = generateTRAGraph(patient=patient, data_path=data_path, output_path='', connected_only=True, plot_graph=False)
        dataset.loc[patient, 'connected_components'] = len(list(nx.connected_component_subgraphs(g)))
        # the max of the number of vertex of the connected components of the graph
        if len(list(nx.connected_component_subgraphs(g))) > 0:
            dataset.loc[patient, 'connected_components_max_size'] = np.max(
                [len(list(component.nodes())) for component in nx.connected_component_subgraphs(g)])
        else:
            dataset.loc[patient, 'connected_components_max_size'] = 0
        for edge in edge_list:
            edge = edge.split(' ')
            if edge[0]  in ['X', 'Y'] and edge[1] in ['X','Y']:
                edge_column = '(' + 'X' + ',' + 'Y' + ')'
            elif edge[0] in ['X', 'Y']:
                edge_column = '(' + edge[1] + ',' + edge[0] + ')'
            elif edge[1] in ['X', 'Y']:
                edge_column = '(' + edge[0] + ',' + edge[1] + ')'
            elif int(edge[0]) < int(edge[1]):
                edge_column = '(' + edge[0] + ',' + edge[1] + ')'
            else:
                edge_column = '(' + edge[1] + ',' + edge[0] + ')'
            edge_weight = int(edge[2])
            # print edge, edge_column, edge_weight
            dataset.loc[patient, edge_column] = edge_weight
        i += 1
    # initialize the chromosome columns at 0
    for chrom in chromosomes:
        dataset['chr_' + str(chrom)] = 0
        dataset['DEL_' + str(chrom)] = 0
        dataset['DUP_' + str(chrom)] = 0

    # initialize the svclass columns at 0
    for cls in svclass:
        dataset[cls] = 0

    # for all patients on the dataset load its breaks and extract their data
    for patient in dataset.index:
        patient_path = data_path + '/raw_original_data/allfiles/' + patient + '.vcf.tsv'
        patient_breaks = pd.read_csv(patient_path, sep='\t', index_col=None)

        # load the chromosomes as strings
        patient_breaks['chrom2'] = patient_breaks['chrom2'].map(str)
        # generate a crosstab of the svclass with the chromosomes
        ct = pd.crosstab(patient_breaks['chrom2'], patient_breaks['svclass'])
        ct.index = ct.index.map(str)
        # print(ct)
        for chrom in chromosomes:
            if chrom in ct.index:
                if 'DEL' in ct.columns:
                    dataset.loc[patient, ['DEL_' + str(chrom)]] = ct.loc[chrom, ['DEL']].values[0]
                if 'DUP' in ct.columns:
                    dataset.loc[patient, ['DUP_' + str(chrom)]] = ct.loc[chrom, ['DUP']].values[0]


        number_of_breaks = len(patient_breaks)
        dataset.loc[patient, 'number_of_breaks'] = number_of_breaks

        # I count how many times appears on the breaks each of the chromosomes.
        contained_chromosomes = patient_breaks[['#chrom1', 'chrom2']].apply(pd.Series.value_counts)
        contained_chromosomes = contained_chromosomes.fillna(0)
        contained_chromosomes[['#chrom1', 'chrom2']] = contained_chromosomes[['#chrom1', 'chrom2']].astype(int)
        contained_chromosomes['chromosome'] = contained_chromosomes.index
        contained_chromosomes['count'] = contained_chromosomes['#chrom1'] + contained_chromosomes['chrom2']
        # Then saves it on the chromosome feature.
        for chrom in contained_chromosomes.index:
            dataset.loc[patient, ['chr_' + str(chrom)]] = contained_chromosomes.loc[chrom, ['count']].values[0]

        # Counts how many breaks of each class there are on the breaks and saves it.
        count_svclass = patient_breaks[['svclass', ]].apply(pd.Series.value_counts)
        for svclass in count_svclass.index:
            dataset.loc[patient, [svclass]] = count_svclass.loc[svclass, ['svclass']].values[0]
    print(output_path + '/' + name)
    dataset.to_csv(output_path +'/'+ name)




def main():
    NUMBER_OF_SAMPLES = -1
    # Data paths:
    DATA_PATH = '../../data_chromosome'
    OUTPUT_PATH = DATA_PATH + '/datasets'
    try: os.mkdir(DATA_PATH)
    except: pass
    try: os.mkdir(OUTPUT_PATH)
    except: pass

    # Directory containing the files
    data_path = DATA_PATH + '/raw_original_data/allfiles'
    all_patients = os.listdir(data_path)[:NUMBER_OF_SAMPLES]
    all_patients = [p.replace('.vcf.tsv','') for p in all_patients]
    name = 'dataset_' + str(NUMBER_OF_SAMPLES) + '_chrom.csv'
    generate_dataset(patients=all_patients, data_path=DATA_PATH, output_path=OUTPUT_PATH, name=name)


if __name__ == '__main__':
    init = time.time()
    main()
    print'Total time:', timedelta(seconds=time.time() - init)
