"""
Goal Dataset:
patient_id | subgraph0 | subgraph1 | ... | subgraphN | metadata | type of cancer (target)

To construct this I need:
subgraphID (0) patients that contain it [patientid0 patientid1 patientidK] Total support of graph0
subgraphID (1) patients that contain it [patientid0 patientid1 patientidK] Total support of graph0
int               array(string)                                              (int)

Subgraph_instance:
    id: int
    patients: array(strings)
    support: int

Data:
    array(Graph)

Generate all data -> order it by support -> select a subset of this graphs -> generate dataset.

For every patient:
        1. Gets the patient graph and plots it.
        2. Gets the all the subgraph patterns of this graph.
        3. Generates the sample in terms of subgraphs of this patient.
            (dict\[sub_graph\] = number of sub_graphs of this type)
        4. Add to the general list the new subgraphs.
    Generates a list of the most common patterns for all the patients.

[1] https://github.com/betterenvi/gSpan

TODO:
    - Ask Dario what should I do with the incongruent data:
    The gspan code returns two different sets of subgraphs if you call the repport (gs._repport_df) or if you load de
    subgraphs from the class. (gs.graphs)
    The first one is the only one that is congruent with the logs, so for now I'm using it.
    - Paralelise the code for generating the dataset ( if possible)
    - Decide if use min support 1 or min support 2
        If i use support 1 it generates a non reasonable number of subgraphs,
        for now i'm testing using min support per patient 2
    - Standarizar el output.
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

from gspan_mining.config import parser
from gspan_mining.gspan import gSpan

from datetime import timedelta
import time
import numpy as np
import pickle
import pandas as pd
import shlex

# This variables are global to simplify testing the code, will be removed later:

NUMBER_OF_SAMPLES = 5

MIN_SUPPORT = 0.8

MAX_DISTANCE = 1500

REPLACE = True

# Data paths:

DATAPATH = '../../data'

GSPAN_DATA_FOLDER = '/all_files_gspan_' + str(MAX_DISTANCE) + '/'

PROCESSED_PATH = DATAPATH + '/tests/processsed_' + str(NUMBER_OF_SAMPLES) + \
                 '_' + str(MIN_SUPPORT) + '_' + str(MAX_DISTANCE) + '.txt'

try:
    os.mkdir(DATAPATH + GSPAN_DATA_FOLDER)
except:
    pass
try:
    os.mkdir(DATAPATH + '/tests')
except:
    pass


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
            support: int
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

    def purge_less_common_subgraphs(self,min):
        self.all_subgraphs = [graph for graph in self.all_subgraphs if graph.support > min]
        to_mantain = [graph.description for graph in self.all_subgraphs]
        for i in range(len(self.patients)):
            new = {key:val for key, val in self.patients[i].graphs.items() if key in to_mantain}
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


def old_generate_subgraphs(gspan_file_name, s, l=3, plot=False):
    filepath = DATAPATH + GSPAN_DATA_FOLDER + gspan_file_name + '.txt'
    args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' ' + '-p ' + str(plot) + ' ' + filepath
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = gSpan(
        database_file_name=FLAGS.database_file_name,
        min_support=FLAGS.min_support,
        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
        max_num_vertices=FLAGS.upper_bound_of_num_vertices,
        max_ngraphs=FLAGS.num_graphs,
        is_undirected=(not FLAGS.directed),
        verbose=FLAGS.verbose,
        visualize=FLAGS.plot,
        where=FLAGS.where
    )

    gs.run()
    report = gs._report_df
    gs = None
    return report


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
            broken_patients = open('logs/broken.txt','w')
            broken_patients.write(path)
    # report.to_csv('isitworking.csv')
    return report


def generate_subgraphs(gspan_file_name, s):
    filepath = DATAPATH + GSPAN_DATA_FOLDER + gspan_file_name + '.txt'
    outputpath = DATAPATH + '/patients_gspan_graphs/'
    try:
        os.mkdir(outputpath)
    except:
        pass
    outputfile = DATAPATH + '/patients_gspan_graphs/' + gspan_file_name

    command = shlex.split("/home/raquel/Documents/DataMining-gSpan/build/gbolt -input_file " + filepath +
                          " -pattern 1 -support " + str(
        s) + " -output_file " + outputfile)
    # I want no output from gspan
    p = subprocess.call(command,stderr=open(os.devnull, 'wb'))
    report = load_report(outputfile, 8)
    return report


def process_patient(patient_id, max_distance, min_support, plot_graph=False):
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

    print 'patient:', patient_id

    # Todo: reuse the data

    if os.path.isfile(DATAPATH + GSPAN_DATA_FOLDER + patient_id + '.txt'):
        report = generate_subgraphs(patient_id, s=min_support)
    else:
        generate_one_patient_graph(patient_id, max_distance, gspan_path=GSPAN_DATA_FOLDER, with_vertex_weight=False,
                                   with_vertex_chromosome=False,
                                   with_edge_weight=False)
        report = generate_subgraphs(patient_id, s=min_support)

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


def process_list_of_patients(patients, max_distance, min_support):
    data = Data()
    print 'number of patients: ', len(patients)
    f = open(PROCESSED_PATH, 'w')
    i = 0
    for patient_id in patients:
        f.write(str(i) + ' ' + patient_id)
        print(str(i) + ' ' + patient_id)
        f.write('\n')
        subgraphs, patient_instance = process_patient(patient_id, max_distance, min_support=min_support)
        data.add_patient(patient_instance)
        f.write('subgraphs: ' + str(len(subgraphs)))
        f.write('\n')
        for graph in subgraphs:
            data.add_subgraph(graph)
        i += 1

    data.print_all()

    file_path = DATAPATH + '/tests' + '/data_' + str(len(patients)) + '_' + str(min_support) + '_' + str(max_distance) + '.pkl'
    Data().save_to_file(file_path, data)

    return file_path


def generate_dataset(path, name='classification_csv'):
    """
    This function generates a dataset using the subgraphs of the patients and the metadata.
    :param path:
    :return:
    """

    metadata = pd.read_csv('../../data/clean_metadata.csv')
    metadata = metadata.set_index('sampleID')

    data = Data().load_from_file(path)
    data.sort_by_support()
    data.purge_less_common_subgraphs(20)

    patients_id = [p.id for p in data.patients]
    selected_patients_metadata = metadata.loc[metadata.index.isin(patients_id)]
    graph_columns = ['graph_' + str(graph.id) for graph in data.all_subgraphs]
    all_columns = np.append(metadata.columns, graph_columns).flatten()

    graphs_dataset = pd.DataFrame(columns=all_columns)
    graphs_dataset = pd.concat([graphs_dataset, selected_patients_metadata])
    # Put 0 in all the columns of this patient
    graphs_dataset.loc[:, graph_columns] = 0
    i = 0
    for patient in data.patients:
        if patient.id in metadata.index:
            if i %100 ==0:
                print i
            for graph_description in patient.graphs.keys():
                id = data.existing_subgraphs[graph_description]
                column = 'graph_' + str(id)
                # Put the support of the graph corresponding to this patient
                graphs_dataset.loc[patient.id, column] = patient.graphs[graph_description]
            i +=1
        else:
            print(patient.id)

    # print(graphs_dataset.head)
    graphs_dataset.to_csv(DATAPATH + '/' + name)


def main():
    # Directory containing the files
    data_path = DATAPATH + '/allfiles'
    all_patients = os.listdir(data_path)[:NUMBER_OF_SAMPLES]
    file_path = process_list_of_patients(all_patients, max_distance=MAX_DISTANCE, min_support=MIN_SUPPORT)
    name = 'classification_dataset_' + str(NUMBER_OF_SAMPLES) + '_' + str(MIN_SUPPORT) + '_' + str(MAX_DISTANCE)
    generate_dataset(file_path, name)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
