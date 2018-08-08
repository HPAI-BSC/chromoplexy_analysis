"""
For every patient:
        1. Gets the patient graph and plots it.
        2. Gets the all the subgraph patterns of this graph.
        3. Generates the sample in terms of subgraphs of this patient.
            (dict\[sub_graph\] = number of sub_graphs of this type)
        4. Add to the general list the new subgraphs.
    Generates a list of the most common patterns for all the patients.
TODO: find a way to save the graph support
[1] https://github.com/betterenvi/gSpan
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

DATAPATH = '../../data'
PATIENT_PATH = DATAPATH + '/one_patient_test'

try:
    os.mkdir(DATAPATH + '/one_patient_test')
except:
    pass


class Data(object):
    """
    This is the data i'll use to generate the features.
    """

    def __init__(self):
        """
        Atributes:
            all_subgraphs: dict[int] = gspan_graph. Dictionary of all the subgraphs.
                :key id:int the global graph id.
                :value subgraph: gspan_graph the subgraph in gspan format.
            all_supports: dict[int] = gspan_graph. Dictionary of all the supports.
                :key id:int the global graph id.
                :value support:int The support of the subgraph correspondent to this id.
        """
        self.all_subgraphs = dict()
        self.all_supports = dict()

    def get_graph_id(self, target_graph):
        try:
            id = self.all_subgraphs.keys()[self.all_subgraphs.values().index(target_graph)]
        except:
            id = len(self.all_subgraphs.keys())
        return id

    def add_subgraph(self, subgraph, support):
        id = self.get_graph_id(subgraph)
        try:
            self.all_subgraphs[id] = subgraph
        except:
            pass
        try:
            self.all_supports[id] += support
        except:
            self.all_supports[id] = support

    def print_all(self):
        for id in self.all_subgraphs.keys():
            print id, self.all_supports[id]
            self.all_subgraphs[id].display()


def generate_subgraphs(gspan_file_name, l=3, s=1, plot=False):
    filepath = DATAPATH + '/allfiles_gspan_format/' + gspan_file_name + '.txt'
    args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' ' + filepath
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = gspanmain(FLAGS)
    supports = gs._report_df['support']
    i = 0
    graph_support = {}
    for support in supports:
        print i, support
        graph_support[i] = support
        i +=1
    if plot:
        for g in gs.graphs.values():
            g.plot()
    return gs, graph_support


def process_patient(patient_id, plot_graph=False, max_distance=1000):
    if plot_graph:
        plot_one_file(patient_id)
    generate_one_patient_graph(patient_id, max_distance, with_vertex_weight=False, with_vertex_chromosome=True,
                               with_edge_weight=False)
    print 'subgraphs of this patient'
    subgraphs = generate_subgraphs(patient_id, plot=False)
    supports = []
    # TODO: find a way to get the supports
    print subgraphs.graphs
    # for id, graph in subgraphs.graphs.items():
    #     print id , graph
    # print(subgraphs)
    return subgraphs, supports


def process_list_of_patients(patients, max_distance=1000):
    data = Data()
    for patient in patients:
        subgraphs = process_patient(patient, max_distance)
        # for subgraph in subgraphs.graphs:
        #     print subgraph, subgraphs._get_support(subgraphs.graphs)
        # data.add_subgraph(subgraphs[subgraph],subgraphs.support)
    data.print_all()


def main():
    test_0 = 'e84e0649-a2e8-4873-9cb6-1aa65601ae3a.vcf.tsv'
    test_1 = '0a9c9db0-c623-11e3-bf01-24c6515278c0.vcf.tsv'
    process_list_of_patients([test_0, test_1])


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
