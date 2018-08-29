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
import subprocess
import sys

sys.path.insert(1, '../../src')

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from step1.main_graph_printer import plot_one_file
from step2.main_gspan_subgraph_generator import generate_one_patient_graph

from datetime import timedelta
import time
import numpy as np
import pickle
import pandas as pd
import shlex


class Subgraph_instance(object):
    """
    This is the class that saves the subgraph data.
    .
    """

    def __init__(self, description=''):
        """
        Atributes:
            id: int
            description: string  ('v 0 1 v 2 8 e 0 2 ')
            patients: array(string)
            support: int  (This variable is the global support of the graph, i.e. respect all the patients)
        """
        self.id = 0
        self.description = description
        self.patients = np.array([])
        self.support = 0

    def add_patients(self, new_patients):
        self.patients = np.append(self.patients, new_patients)

    def add_support(self, new_support):
        self.support += new_support

    def print_subgraph(self):
        print 'id: ', self.id
        print 'description: ', self.description
        print 'support: ', self.support
        # print 'patients: ', self.patients


class Patient_instance(object):
    """
        This is the class that saves the subgraph data.
        .
        """

    def __init__(self, id):
        """
        Atributes:
            id: string
            graphs: dict(description(string):support(int))
        """
        self.id = id.replace('.vcf.tsv', '')
        self.graphs = {}

    def add_graph(self, new_graph_description, new_graph_support):
        self.graphs[new_graph_description] = new_graph_support

    def print_patient(self):
        print 'id: ', self.id
        print 'graphs: ', self.graphs


class Data(object):
    """
    This is the data i'll use to generate the features.
    """

    def __init__(self):
        """
        Atributes:
            all_subgraphs: array(Graph_instance)
            existing_subgraphs = key: description value: id
            patients = array(Patient_Instance)
        """
        self.all_subgraphs = []
        self.existing_subgraphs = {}
        self.patients = []

    def add_subgraph(self, new_graph):
        """
        I add the new graph to the dataset or actualize it if the graph is already there.
        :param new_graph: Graph_instance
        :return:
        """
        if new_graph.description in self.existing_subgraphs.keys():
            # if I alredy have this graph I actualize the data (adding the patients and the support to this graph)
            id = self.existing_subgraphs[new_graph.description]
            self.all_subgraphs[id].add_patients(new_graph.patients)
            self.all_subgraphs[id].add_support(new_graph.support)
        else:
            # if I don't have the graph I give it an id and I add it to the dataset
            id = len(self.existing_subgraphs)
            new_graph.id = id
            self.all_subgraphs.append(new_graph)
            self.existing_subgraphs[new_graph.description] = new_graph.id

    def add_patient(self, newPatient):
        self.patients.append(newPatient)

    def sort_by_support(self):
        import operator
        self.all_subgraphs.sort(key=operator.attrgetter('support'), reverse=True)
        # for i in range(len(self.all_subgraphs)):
        #     self.all_subgraphs[i].id = i
        #     self.existing_subgraphs[self.all_subgraphs[i].description] = i

    def print_all(self):
        print 'Report of data:'
        print 'number of subgraphs', len(self.all_subgraphs)
        print 'number of patients', len(self.patients)
        for graph in self.all_subgraphs[:5]:
            graph.print_subgraph()

    def print_most_common(self, number_of_graphs):
        print 'number of subgraphs', len(self.all_subgraphs)
        for graph in self.sort_by_support()[:number_of_graphs]:
            graph.print_subgraph()

    def purge_less_common_subgraphs(self, min):
        self.all_subgraphs = [graph for graph in self.all_subgraphs if graph.support > min]
        to_mantain = [graph.description for graph in self.all_subgraphs]
        for i in range(len(self.patients)):
            new = {key: val for key, val in self.patients[i].graphs.items() if key in to_mantain}
            self.patients[i].graphs = None
            self.patients[i].graphs = new

    @staticmethod
    def save_to_file(filepath, dataobject):
        fileObject = open(filepath, 'wb')
        pickle.dump(dataobject, fileObject)

    @staticmethod
    def load_from_file(filepath):
        fileObject = open(filepath, 'rb')
        dataobject = pickle.load(fileObject)
        return dataobject


def load_report(path, cores):
    """
    The report is contained on a set of files, one per core used to mine the graphs.
    Every file contains subgraphs with the next extructure:

    t # 0 * 2
    v 0 2
    v 1 2
    e 0 1 2
    x: 0 1
    :param path:
    :param cores:
    :return:
    """
    report = pd.DataFrame(columns=['description', 'support'])
    index = 0
    support = 0
    description = ''
    for core in range(cores):
        try:
            f = open(path + '.t' + str(core), 'r')
            for l in f:
                # Skip empty lines
                if l == '\n':
                    continue
                # If its a new graph. Store the current one and initialize the next
                if l[0] == 't':
                    # Obtain the support of this graph
                    support = int(l.split()[-1])
                    description = ''
                    # Initialize next graph
                    first_edge = True
                # Its a new vertex. Add it.
                if l[0] == 'v':
                    description += l.replace('\n', ' ')
                # Its a new edge. Add it.
                if l[0] == 'e':
                    description += l.replace('\n', ' ')
                # Its the graph support. Store it.
                # If the graph is finished sum one to the index
                if l[0] == 'x':
                    report.loc[index] = [description, support]
                    index += 1
            # When done with this file i remove it from disk
            os.remove(path + '.t' + str(core))
        except:
            broken_patients = open('logs/broken.txt', 'w')
            broken_patients.write(path)
    # report.to_csv('isitworking.csv')
    return report


def generate_subgraphs(gspan_file_name, s, data_path, gspan_data_folder):
    filepath = data_path + gspan_data_folder + gspan_file_name + '.txt'
    outputpath = data_path + '/temporal_gspan_data/'
    try:
        os.mkdir(outputpath)
    except:
        pass
    outputfile = data_path + '/temporal_gspan_data/' + gspan_file_name

    command = shlex.split("/home/raquel/Documents/DataMining-gSpan/build/gbolt -input_file " + filepath +
                          " -pattern 1 -support " + str(
        s) + " -output_file " + outputfile)
    # I want no output from gspan
    p = subprocess.call(command, stderr=open(os.devnull, 'wb'))
    report = load_report(outputfile, 8)
    return report


def process_patient(patient_id, max_distance, min_support, data_path, gspan_data_folder, plot_graph=False):
    """
    This function generates an array of subgraphs instances using the report returned by gspan
    :param patient_id:
    :param max_distance:
    :param min_support:
    :param plot_graph:
    :return:
    """
    if plot_graph:
        plot_one_file(patient_id)

    if os.path.isfile(data_path + gspan_data_folder + patient_id + '.txt'):
        report = generate_subgraphs(patient_id, s=min_support, data_path=data_path, gspan_data_folder=gspan_data_folder)
    else:
        generate_one_patient_graph(patient_id, max_distance, gspan_path=gspan_data_folder, with_vertex_weight=False,
                                   with_vertex_chromosome=False,
                                   with_edge_weight=False)
        report = generate_subgraphs(patient_id, s=min_support, data_path=data_path, gspan_data_folder=gspan_data_folder)

    patient = Patient_instance(patient_id)

    subgraphs = []
    for i in report.index:
        graph_description = report['description'][i]
        graph_support = report['support'][i]

        patient.add_graph(graph_description, graph_support)

        subgraph = Subgraph_instance(graph_description)
        subgraph.add_patients([patient_id])
        subgraph.add_support(graph_support)
        subgraphs.append(subgraph)
    return subgraphs, patient


def process_list_of_patients(patients, max_distance, min_support, data_path, gspan_data_folder, processed_path, ):
    data = Data()
    print 'number of patients: ', len(patients)
    f = open(processed_path, 'w')
    i = 0
    for patient_id in patients:
        f.write(str(i) + ' ' + patient_id)
        # print(str(i) + ' ' + patient_id)
        f.write('\n')
        subgraphs, patient_instance = process_patient(patient_id, max_distance, min_support=min_support,
                                                      data_path=data_path, gspan_data_folder=gspan_data_folder)
        data.add_patient(patient_instance)
        f.write('subgraphs: ' + str(len(subgraphs)))
        f.write('\n')
        for graph in subgraphs:
            data.add_subgraph(graph)
        i += 1

    # data.print_all()

    file_path = data_path + '/raw_processed_data' + '/data_' + str(len(patients)) + '_' + str(min_support) + '_' + str(
        max_distance) + '.pkl'
    Data().save_to_file(file_path, data)

    return file_path


def generate_dataset(path, data_path, name='classification_csv'):
    """
    This function generates a dataset using the subgraphs of the patients and the metadata.
    :param path:
    :return:
    """
    print('Generating csv..')
    metadata = pd.read_csv(data_path + '/raw_original_data/metadatos_v2.0.csv')
    metadata = metadata.set_index('sampleID')

    data = Data().load_from_file(path)
    data.sort_by_support()
    data.purge_less_common_subgraphs(5)

    patients_id = [p.id for p in data.patients]
    selected_patients_metadata = metadata.loc[metadata.index.isin(patients_id)]
    graph_columns = ['graph_' + str(graph.id) for graph in data.all_subgraphs]
    all_columns = np.append(metadata.columns, graph_columns).flatten()

    graphs_dataset = pd.DataFrame(columns=all_columns)
    graphs_dataset = pd.concat([graphs_dataset, selected_patients_metadata])
    # Put 0 in all the columns of this patient
    graphs_dataset.loc[:, graph_columns] = 0
    i = 0
    print 'Patients without metadata: '
    for patient in data.patients:
        if patient.id in metadata.index:
            # if i %100 ==0:
            #     print i
            for graph_description in patient.graphs.keys():
                id = data.existing_subgraphs[graph_description]
                column = 'graph_' + str(id)
                # Put the support of the graph corresponding to this patient
                graphs_dataset.loc[patient.id, column] = patient.graphs[graph_description]
            i += 1
        else:
            # TODO: move this files to another folder instead of printing them
            print(patient.id)

    # I'll try to add chromosome relative information.
    chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']
    graphs_dataset['number_of_breaks'] = 0

    svclass = ['DUP', 'DEL', 'TRA', 'h2hINV', 't2tINV']

    for chrom in chromosomes:
        graphs_dataset['chr_' + str(chrom)] = 0

    for cls in svclass:
        graphs_dataset[cls] = 0

    for patient in graphs_dataset.index:
        patient_path = data_path + '/raw_original_data/allfiles/' + patient + '.vcf.tsv'
        patient_breaks = pd.DataFrame.from_csv(patient_path, sep='\t', index_col=None)
        # print(patient_breaks.columns, patient_breaks.shape)

        number_of_breaks = len(patient_breaks)
        graphs_dataset.loc[patient, 'number_of_breaks'] = number_of_breaks

        contained_chromosomes = patient_breaks[['#chrom1', 'chrom2']].apply(pd.Series.value_counts)
        contained_chromosomes = contained_chromosomes.fillna(0)
        contained_chromosomes[['#chrom1', 'chrom2']] = contained_chromosomes[['#chrom1', 'chrom2']].astype(int)
        contained_chromosomes['chromosome'] = contained_chromosomes.index
        contained_chromosomes['count'] = contained_chromosomes['#chrom1'] + contained_chromosomes['chrom2']
        for chrom in contained_chromosomes.index:
            graphs_dataset.loc[patient, ['chr_' + str(chrom)]] = contained_chromosomes.loc[chrom, ['count']].values[0]

        count_svclass = patient_breaks[['svclass', ]].apply(pd.Series.value_counts)
        for svclass in count_svclass.index:
            graphs_dataset.loc[patient, [svclass]] = count_svclass.loc[svclass, ['svclass']].values[0]

    try:
        graphs_dataset.to_csv(data_path + '/datasets/' + name + '.csv')
    except:
        os.mkdir(data_path + '/datasets/')
        graphs_dataset.to_csv(data_path + '/datasets/' + name + '.csv')
    print('Csv generated')


def main():

    # This variables are global to simplify testing the code, will be removed later:
    NUMBER_OF_SAMPLES = -1

    MAX_DISTANCE = 2000

    supports = [1, 0.9, 0.7, 0.6]

    time_per_support = {}
    for support in supports:
        init = time.time()

        MIN_SUPPORT = support
        # Data paths:
        DATAPATH = '../../data'

        GSPAN_DATA_FOLDER = '/all_files_gspan_' + str(MAX_DISTANCE) + '/'

        PROCESSED_PATH = DATAPATH + '/raw_processed_data/processsed_' + str(NUMBER_OF_SAMPLES) + \
                         '_' + str(MIN_SUPPORT) + '_' + str(MAX_DISTANCE) + '.txt'

        try:
            os.mkdir(DATAPATH + GSPAN_DATA_FOLDER)
        except:
            pass
        try:
            os.mkdir(DATAPATH + '/raw_processed_data')
        except:
            pass

        # Directory containing the files
        data_path = DATAPATH + '/raw_original_data/allfiles'
        all_patients = os.listdir(data_path)[:NUMBER_OF_SAMPLES]
        print('Generating the raw data...')
        # file_path = process_list_of_patients(all_patients, max_distance=MAX_DISTANCE, min_support=MIN_SUPPORT,
        #                                      data_path=DATAPATH, processed_path=PROCESSED_PATH,
        #                                      gspan_data_folder=GSPAN_DATA_FOLDER)
        file_path = '../../data/raw_processed_data/data_2597_' + str(MIN_SUPPORT) + '_2000.pkl'
        name = 'classification_dataset_' + str(NUMBER_OF_SAMPLES) + '_' + str(MIN_SUPPORT) + '_' + str(MAX_DISTANCE) + '_nan'
        try:
            generate_dataset(path=file_path, data_path=DATAPATH, name=name)
        except:
            pass
        end_time = timedelta(seconds=time.time() - init)
        time_per_support[support] = end_time

    for key in time_per_support.keys():
        print 'support:', key
        print 'time:',  time_per_support[key]

if __name__ == '__main__':
    init = time.time()
    main()
    print'Total time:', timedelta(seconds=time.time() - init)
