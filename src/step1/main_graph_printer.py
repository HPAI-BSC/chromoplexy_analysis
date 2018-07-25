'''
Generate a graph from each patient data, and plot it.
'''

from loader import load_breaks
from graph_builder import generateGraph
from graphnx import generateNXGraph, printGraph

import os
import sys

ask_distance = False

if ask_distance:
	if len(sys.argv) != 2:
		raise Exception(
			'This function must be called with one parameter. An integer indicating the length of the sliding window.')
	max_distance = int(sys.argv[1])
else:
	max_distance = 1000

# Directory containing the files
data_path = '../data/allfiles'
def plot_all():
	# Iterate over the files
	for file_name in os.listdir(data_path):
		# Load the brakes
		breaks, list_of_pairs = load_breaks(os.path.join(data_path, file_name))
		if len(breaks) == 0:
			print 'WARNING: Empty data file', file_name
			continue
		# Generate the vertices and edges
		adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs, max_distance)
		# Create the graph
		g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges,vertex_weights, self_links=False, connected_only=True)
		# Print the graph
		print 'Showing graph of ', file_name
		printGraph(g,show_vertex_weights=False)

		print vertex_ranges


def plot_one_file(file_name):
	# Load the brakes
	breaks, list_of_pairs = load_breaks(os.path.join(data_path, file_name))
	if len(breaks) == 0:
		print 'WARNING: Empty data file', file_name
	# Generate the vertices and edges
	adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights = generateGraph(breaks, list_of_pairs, max_distance)
	# Create the graph
	g = generateNXGraph(adjacency_matrix, vertex_labels, vertex_ranges, vertex_weights, self_links=False,
						connected_only=True)
	# Print the graph
	print 'Showing graph of ', file_name
	printGraph(g, name=file_name,visualize=False, show_vertex_weights=False)

	# print vertex_ranges

def main():
	test_file = 'e84e0649-a2e8-4873-9cb6-1aa65601ae3a.vcf.tsv'
	plot_one_file(test_file)


if __name__ == '__main__':
	import time
	from datetime import  timedelta
	init = time.time()
	main()
	print('time:', timedelta(seconds=time.time() - init))
