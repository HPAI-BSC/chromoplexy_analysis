"""
For every patient:
        1. Gets the patient graph and plots it.
        2. Gets the all the subgraph patterns of this graph.
        3. Generates the sample in terms of subgraphs of this patient.
            (dict\[sub_graph\] = number of sub_graphs of this type)
        4. Add to the general list the new subgraphs.
    Generates a list of the most common patterns for all the patients.
"""


import os
import sys

sys.path.insert(1, '../src')

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

def generate_subgraphs(gspan_file_name, l=3,s=1, plot=False):
	filepath = DATAPATH + '/allfiles_gspan_format/'+ gspan_file_name + '.txt'
	args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' ' +filepath
	FLAGS, _ = parser.parse_known_args(args=args_str.split())
	gs =  gspanmain(FLAGS)
	if plot:
		for g in gs.graphs.values():
			g.plot()
	return gs


def process_patient(patient_id, plot_graph=False, max_distance=1000):
	if plot_graph:
		plot_one_file(patient_id)
	generate_one_patient_graph(patient_id, max_distance,with_vertex_weight=False,with_vertex_chromosome=True,with_edge_weight=False)
	subgraphs = generate_subgraphs(patient_id,plot=True)



def main():
	test_file = 'e84e0649-a2e8-4873-9cb6-1aa65601ae3a.vcf.tsv'
	process_patient(test_file)

if __name__ == '__main__':
	init = time.time()
	main()
	print'time:', timedelta(seconds=time.time() - init)
