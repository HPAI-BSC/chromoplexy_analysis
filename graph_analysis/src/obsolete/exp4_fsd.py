'''
Runs the gSpan algorithm to find frequent subgraphs
'''

import os
import sys

from cStringIO import StringIO

old_stdout = sys.stdout
redirected_output = sys.stdout = StringIO()


def call_gspan_from_library(filepath, outputpath):
    # args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' -u 4 -v False ' + '-p ' + str(plot) + ' ' + filepath
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    args_str = filepath + ' -l 3 ' + '-s 5000'
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
    with open(outputpath.replace('.txt', '_frequent_subgraphs.txt'), 'w') as f:
        f.write(redirected_output.getvalue())
    sys.stdout = old_stdout


def call_gspan_locally(filepath, outputpath, gspan_main_path):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    sys.argv = [gspan_main_path, filepath, '-l 3', '-s 5000']
    try:
        execfile(gspan_main_path)
    except:
        raise
    with open(outputpath.replace('.txt', '_frequent_subgraphs.txt'), 'w') as f:
        f.write(redirected_output.getvalue())
    f.close()
    sys.stdout = old_stdout


def run_gspan(file_path):
    try:
        from gspan_mining.config import parser
        from gspan_mining.gspan import gSpan
        call_gspan_from_library(file_path, file_path)
    except:
        gspan_main_path = ''
        call_gspan_locally(data_path + file_name, data_path + file_name, gspan_main_path)


# sys.path.insert(0, '/home/dariog/repos/gSpan1')
data_path = '../data/allfiles_gspan/'
files = os.listdir(data_path)
test_files = ['gspan_subgraphs_w100.txt']

for file_name in test_files:
    run_gspan(data_path + file_name)
