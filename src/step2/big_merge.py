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
TODO: ask Dario what should I do with the incongruent data:
    The gspan code returns two different sets of subgraphs if you call the repport (gs._repport_df) or if you load de
    subgraphs from the class. (gs.graphs)
    The first one is the only one that is congruent with the logs, so for now I'm using it.
"""

import os
import sys

sys.path.insert(1, '../../src')

from step1.loader import load_breaks
from step1.graph_builder import generateGraph
from step1.graphnx import generateNXGraph
from step1.main_graph_printer import plot_one_file
from step2.main_gspan_subgraph_generator import generate_one_patient_graph

import networkx as nx
from gspan_mining.config import parser
from gspan_mining.main import main as gspanmain

from datetime import timedelta
import time
import numpy as np

DATAPATH = '../../data'
PATIENT_PATH = DATAPATH + '/one_patient_test'

try:
    os.mkdir(DATAPATH + '/one_patient_test')
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
        print 'subgraph report: '
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
            print 'YAYAYAYAYAYAY'
            id = self.existing_subgraphs[new_graph.description]
            self.all_subgraphs[id].add_patients(new_graph.patients)
            self.all_subgraphs[id].add_support(new_graph.support)
        else:
            # if I don't have the graph I give it an id and I add it to the dataset
            id = len(self.existing_subgraphs)
            new_graph.id = id
            self.all_subgraphs.append(new_graph)
            self.existing_subgraphs[new_graph.description] = new_graph.id

    def print_all(self):
        print 'Report of data:'
        print 'number of subgraphs', len(self.all_subgraphs)
        for graph in self.all_subgraphs:
            graph.print_subgraph()


def generate_subgraphs(gspan_file_name, l=3, s=1, plot=False):
    filepath = DATAPATH + '/allfiles_gspan_format/' + gspan_file_name + '.txt'
    args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' ' + '-p ' + str(plot) + ' ' + filepath
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = gspanmain(FLAGS)
    return gs._report_df


def process_patient(patient_id, max_distance=1000, plot_graph=False,):
    if plot_graph:
        plot_one_file(patient_id)
    generate_one_patient_graph(patient_id, max_distance, with_vertex_weight=False, with_vertex_chromosome=True,
                               with_edge_weight=False)
    print 'subgraphs of this patient'
    report = generate_subgraphs(patient_id, plot=False)
    subgraphs = []
    for i in report.index:
        graph_description = report['description'][i]
        graph_support = report['support'][i]
        subgraph = Subgraph_instance(graph_description)
        subgraph.add_patients([patient_id])
        subgraph.add_support(graph_support)
        subgraph.print_subgraph()
        subgraphs.append(subgraph)
    return subgraphs


def process_list_of_patients(patients, max_distance=1000):
    data = Data()
    for patient in patients:
        subgraphs = process_patient(patient, max_distance)
        for graph in subgraphs:
            data.add_subgraph(graph)
    data.print_all()


def main():
    test_0 = 'e84e0649-a2e8-4873-9cb6-1aa65601ae3a.vcf.tsv'
    test_1 = '0a9c9db0-c623-11e3-bf01-24c6515278c0.vcf.tsv'
    process_list_of_patients([test_1,test_0])


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
