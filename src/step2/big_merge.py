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
"""

import os
import sys

sys.path.insert(1, '../../src')

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


from step1.loader import load_breaks
from step1.graph_builder import generateGraph
from step1.graphnx import generateNXGraph
from step1.main_graph_printer import plot_one_file
from step2.main_gspan_subgraph_generator import generate_one_patient_graph

from gspan_mining.config import parser
from gspan_mining.main import main as gspanmain

from gspan_mining.gspan import  gSpan

from datetime import timedelta
import time
import numpy as np
import pickle


DATAPATH = '../../data'

MAX_DISTANCE = 2000

GSPAN_DATA_FOLDER = '/all_files_gspan_' + str(MAX_DISTANCE) + '/'

try:
    os.mkdir(DATAPATH + GSPAN_DATA_FOLDER)
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
        print 'patients: ', self.patients


class Data(object):
    """
    This is the data i'll use to generate the features.
    """

    def __init__(self):
        """
        Atributes:
            all_subgraphs: array(Graph_instance)
            existing_subgraphs = key: description value: id 
        """
        self.all_subgraphs = []
        self.existing_subgraphs = {}

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

    def sort_by_support(self):
        import operator
        sorted_x = sorted(self.existing_subgraphs, key=operator.attrgetter('support'))
        return sorted_x

    def print_all(self):
        print 'Report of data:'
        print 'number of subgraphs', len(self.all_subgraphs)
        for graph in self.all_subgraphs[:10]:
            graph.print_subgraph()

    def print_most_common(self, number_of_graphs):
        print 'number of subgraphs', len(self.all_subgraphs)
        for graph in self.sort_by_support()[:number_of_graphs]:
            graph.print_subgraph()

    @staticmethod
    def save_to_file(filepath, dataobject):
        fileObject = open(filepath, 'wb')
        pickle.dump(dataobject, fileObject)

    @staticmethod
    def load_from_file(filepath):
        fileObject = open(filepath, 'rb')
        dataobject = pickle.load(fileObject)
        return dataobject


def generate_subgraphs(gspan_file_name, l=3, s=2, plot=False):
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


def process_patient(patient_id, max_distance=1000, plot_graph=False, ):
    if plot_graph:
        plot_one_file(patient_id)

    print 'subgraphs of this patient'
    try:
        report = generate_subgraphs(patient_id, plot=False)
    except:
        generate_one_patient_graph(patient_id, max_distance, gspan_path=GSPAN_DATA_FOLDER, with_vertex_weight=False,
                                   with_vertex_chromosome=False,
                                   with_edge_weight=False)
        report = generate_subgraphs(patient_id, plot=False)

    subgraphs = []
    for i in report.index:
        graph_description = report['description'][i]
        graph_support = report['support'][i]
        subgraph = Subgraph_instance(graph_description)
        subgraph.add_patients([patient_id])
        subgraph.add_support(graph_support)
        subgraphs.append(subgraph)
    return subgraphs


def process_list_of_patients(patients, max_distance=1000):
    data = Data()
    print 'number of patients: ', len(patients)
    f = open('processsed.txt', 'w')
    i = 0
    for patient in patients:
        f.write(str(i) + ' ' + patient)
        f.write('\n')
        subgraphs = process_patient(patient, max_distance)
        f.write('subgraphs: ' + str(len(subgraphs)))
        f.write('\n')
        for graph in subgraphs:
            data.add_subgraph(graph)
        i += 1

    data.print_all()
    Data().save_to_file('data_50_1_' + str(max_distance) + '.pkl',data)


def test():
    test_0 = 'e84e0649-a2e8-4873-9cb6-1aa65601ae3a.vcf.tsv'
    test_1 = '0a9c9db0-c623-11e3-bf01-24c6515278c0.vcf.tsv'
    conflictive = 'b8f3137e-5e92-4a56-90d4-884a4ed2ef9c.vcf.tsv'
    conf = 'd60f880a-c622-11e3-bf01-24c6515278c0.vcf.tsv'
    all_patients = [conf]
    process_list_of_patients(all_patients)


def main():
    # Directory containing the files
    data_path = DATAPATH + '/allfiles'
    all_patients = os.listdir(data_path)
    process_list_of_patients(all_patients)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
