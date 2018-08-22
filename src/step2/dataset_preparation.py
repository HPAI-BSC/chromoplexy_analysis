"""
This function generates the dataset for classification using the clean metadata and the subgraphs generated.

"""

import pandas as pd
import sys

sys.path.insert(1, '../../src')

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from datetime import timedelta
import time

from step2.graph_per_patient_processing import Data


# Data paths:

DATAPATH = '../../data'


def generate_dataset(path):
    """
    This function generates a dataset using the subgraphs of the patients and the metadata.
    Todo: move it to a new file.
    :param path:
    :return:
    """
    data = Data().load_from_file(path)
    data.sort_by_support()
    data.print_all()

    metadata = pd.read_csv('../../data/clean_metadata.csv')

    print(metadata.head())

    print(metadata.colums())

    graph_columns = ['graph_' + str(i) for i in range(len(data.all_subgraphs))]


    graphs_dataset = pd.DataFrame()


def main():
    path = DATAPATH + '/tests/data_100_1_2000.pkl'
    generate_dataset(path)

    
if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
