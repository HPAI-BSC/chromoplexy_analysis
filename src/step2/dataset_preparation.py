"""
This function generates the dataset for classification using the clean metadata and the subgraphs generated.

"""

import sys

sys.path.insert(1, '../../src')

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from datetime import timedelta
import time
import pandas as pd
import numpy as np

from step2.graph_per_patient_processing import Patient_instance, Subgraph_instance, Data


# Data paths:

DATAPATH = '../../data'


def generate_dataset(path):
    """
    This function generates a dataset using the subgraphs of the patients and the metadata.
    :param path:
    :return:
    """

    metadata = pd.read_csv('../../data/clean_metadata.csv')
    metadata = metadata.set_index('sampleID')

    data = Data().load_from_file(path)
    data.sort_by_support()

    patients_id = [p.id for p in data.patients]
    selected_patients_metadata = metadata.loc[metadata.index.isin(patients_id)]
    graph_columns = ['graph_' + str(graph.id) for graph in data.all_subgraphs]
    all_columns = np.append(metadata.columns, graph_columns).flatten()

    graphs_dataset = pd.DataFrame(columns=all_columns)
    graphs_dataset = pd.concat([graphs_dataset, selected_patients_metadata])
    graphs_dataset.loc[:, graph_columns] = 0

    for patient in data.patients:
        # Put 0 in all the columns of this patient
        for graph_description in patient.graphs.keys():
            id = data.existing_subgraphs[graph_description]
            column = 'graph_' + str(id)
            # Put the support of the graph corresponding to this patient
            graphs_dataset.loc[patient.id, column] = patient.graphs[graph_description]

    print(graphs_dataset.head)
    graphs_dataset.to_csv(DATAPATH + '/classification_dataset.csv')


def main():
    path = DATAPATH + '/tests/data_100_1_2000.pkl'
    generate_dataset(path)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
