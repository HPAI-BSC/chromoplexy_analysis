"""
The objective of this code is to analyze the traslocation graphs.

Process:
    - Generate the gspan compatible graphs 'v vid vertex_label' 'e vid0 vid1 edge_label' (done)
    - Connected component histogram + mean size between components -> to features (done)
    - Centrality study -> Ranking of the most central vertex -> Feature: central_rank_chrom_1 = [1,23] rank (done)
    - Number of triangles per patient (gspan)
    - Gspan (less than 4 vertex)
"""

import sys
import pickle

sys.path.insert(1, '../../src')

import pandas as pd
from natsort import natsorted
import networkx as nx
from matplotlib import pyplot as plt
import gc
import numpy as np
import os
from datetime import timedelta
import time

from step2.main_gspan_subgraph_generator import get_traslocation_graph
from step1.graph_builder import generateGraph, generateTRAGraph
from step2.main_gspan_subgraph_generator import generate_one_patient_graph

from gspan_mining.config import parser
from gspan_mining.gspan import gSpan


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
        """
        Removes the subgraphs that appear less than min on all the dataset.
        :param min:
        :return:
        """
        self.all_subgraphs = [graph for graph in self.all_subgraphs if graph.support >= min]
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

def generate_subgraphs(gspan_file_name, s, data_path, gspan_data_folder):
    """
    This function generates the subgraphs ussing a C implementation of gspan
    :param gspan_file_name:
    :param s:
    :param data_path:
    :param gspan_data_folder:
    :return:
    """
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


def generate_subgraphs_one_core(gspan_file_name, s, gspan_data_folder, l=3, plot=False):
    filepath = gspan_data_folder + gspan_file_name + '.txt'
    args_str = ' -s ' + str(s) + ' -l ' + str(l) + ' -u 4 -v False ' + '-p ' + str(plot) + ' ' + filepath
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


def process_patient_original_graphs(patient_id, max_distance, min_support, data_path, gspan_data_folder, plot_graph=False):
    """
    This function generates an array of subgraphs instances using the report returned by gspan
    :param patient_id:
    :param max_distance:
    :param min_support:
    :param plot_graph:
    :return:
    """

    if not os.path.isfile(data_path + gspan_data_folder + patient_id + '.txt'):
        generate_one_patient_graph(patient_id, max_distance, general_data_path=data_path, gspan_path=gspan_data_folder,
                                   with_vertex_weight=False,
                                   with_vertex_chromosome=False,
                                   with_edge_weight=False)
    try:
        report = generate_subgraphs_one_core(patient_id, s=min_support,
                                             gspan_data_folder=gspan_data_folder)

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
    except Exception as e:
        print(e)
        print(patient_id, 'has no graphs')
        patient = Patient_instance(patient_id)
        subgraphs = []

    return subgraphs, patient


def process_list_of_patients_original_graphs(patients, max_distance, min_support, data_path, gspan_data_folder, processed_path):
    data = Data()
    print 'number of patients: ', len(patients)
    f = open(processed_path, 'w')
    i = 0
    for patient_id in patients:
        f.write(str(i) + ' ' + patient_id)
        # print(str(i) + ' ' + patient_id)
        f.write('\n')
        subgraphs, patient_instance = process_patient_original_graphs(patient_id, max_distance, min_support=min_support,
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


def process_patient_traslocations(patient_id, min_support, data_path, gspan_data_folder):
    """
    This function generates an array of subgraphs instances using the report returned by gspan
    :param patient_id:
    :param max_distance:
    :param min_support:
    :param plot_graph:
    :return:
    """
    if not os.path.isfile(data_path + gspan_data_folder + patient_id + '.txt'):
        get_traslocation_graph(patient_id, data_path, gspan_path=gspan_data_folder)
    try:
        report = generate_subgraphs_one_core(patient_id, s=min_support,
                                             gspan_data_folder=gspan_data_folder)

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
    except Exception as e:
        print(e)
        patient = Patient_instance(patient_id)
        subgraphs = []

    return subgraphs, patient


def process_list_of_patients(patients, min_support, data_path, gspan_data_folder, output_path,processed_path='proc.txt'):
    data = Data()
    print 'number of patients: ', len(patients)
    f = open(processed_path, 'w')
    i = 0
    for patient_id in patients:
        f.write(str(i) + ' ' + patient_id)
        f.write('\n')
        subgraphs, patient_instance = process_patient_traslocations(patient_id, min_support=min_support,
                                                                    data_path=data_path, gspan_data_folder=gspan_data_folder)
        data.add_patient(patient_instance)
        f.write('subgraphs: ' + str(len(subgraphs)))
        f.write('\n')
        for graph in subgraphs:
            data.add_subgraph(graph)
        i += 1

    file_path = output_path + '/data_' + str(len(patients)) + '_' + '.pkl'
    Data().save_to_file(file_path, data)

    return file_path


def generate_dataset_gspan(path, data_path, min_graphs, name='classification_csv'):
    """
    This function generates a dataset using the subgraphs of the patients and the metadata.
    :param path:
    :return:
    """
    print('Generating csv..')

    data = Data().load_from_file(path)
    # data.sort_by_support()
    # data.purge_less_common_subgraphs(min_graphs)
    graph_description_df = pd.DataFrame(columns=['id', 'description'])
    graph_description_df['id'] = 0
    graph_description_df['description'] = ''
    graph_description_df['support'] = 0

    graph_columns = ['graph_' + str(graph.id) for graph in data.all_subgraphs]

    graphs_dataset = pd.DataFrame(columns=graph_columns)
    # Put 0 in all the columns of this patient
    graphs_dataset.loc[:, graph_columns] = 0
    i = 0
    # print 'Patients without metadata: '
    for patient in data.patients:
        # adding gspan graphs
        for graph_description in patient.graphs.keys():
            # print graph_description
            id = data.existing_subgraphs[graph_description]
            column = 'graph_' + str(id)
            graph_description_df.loc[id, 'id'] = id
            graph_description_df.loc[id, 'description'] = graph_description
            graph_description_df.loc[id, 'support'] = data.all_subgraphs[id].support
            # Put the support of the graph corresponding to this patient
            graphs_dataset.loc[patient.id, column] = patient.graphs[graph_description]

        i += 1
    print(graph_description_df)
    try:
        os.mkdir(data_path + '/datasets_only_graphs/')
    except:
        pass

    graph_description_df.to_csv(data_path + '/datasets_only_graphs/' + name + 'graph_desc' + '.csv')
    graphs_dataset.to_csv(data_path + '/datasets_only_graphs/' + name + '.csv')

    print('Csv generated')

def generate_tra_graph_related_dataset(patients, data_path, name='classification_csv'):
    graphs_dataset = pd.DataFrame(columns=['connected_components','connected_components_mean_size'])
    metadata = pd.read_csv(data_path + '/raw_original_data/metadatos_v2.0.csv')
    metadata = metadata.set_index('sampleID')

    for patient in patients:
        if patient in metadata.index:
            # adding traslocation graph info
            graphs_dataset.loc[patient, 'histology'] = metadata.loc[patient, 'histology_tier1']
            g, edge_list, adjacency_matrix_connected_only = generateTRAGraph(patient, data_path,
                                                                             connected_only=True, plot_graph=False)
            graphs_dataset.loc[patient, 'connected_components'] = len(list(nx.connected_component_subgraphs(g)))
            # the mean of the number of vertex of the connected components of the graph
            graphs_dataset.loc[patient, 'connected_components_mean_size'] = np.mean(
                [len(list(component.nodes())) for component in nx.connected_component_subgraphs(g)])

            centrality = nx.degree_centrality(g)
            for node, centr_degree in centrality.items():
                graphs_dataset.loc[patient, str(node) + '_cent'] = centr_degree

            sorted_by_value = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
            if len(sorted_by_value) > 0:
                graphs_dataset.loc[patient, 'most_centered'] = sorted_by_value[0][0]
            i = 1
            for node, centr_degree in sorted_by_value:
                graphs_dataset.loc[patient, str(node) + '_cent_ranking'] = i
                i +=1

    graphs_dataset =graphs_dataset.fillna(0)
    graphs_dataset.to_csv(data_path + '/datasets_only_graphs/' + name + '_tra_graphs.csv')
    print(data_path + '/datasets_only_graphs/' + name + '_tra_graphs.csv')


def plot_graph_info(path):
    histologies = ['ECTODERM','ENDODERM', 'NEURAL_CREST', 'MESODERM']
    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', '20', '21', '22', 'X', 'Y']
    from collections import Counter
    df = pd.read_csv(path)
    subsets_of_df = {}
    for hist in histologies:
        subsets_of_df[hist] = df.loc[df['histology'] == hist]

    for column in df.columns:
        print(column)
        plt.figure(figsize=(20, 10))
        if 'Unnamed' in column:
            pass
        elif 'most_centered' in column:
            values_ECTODERM = Counter(subsets_of_df['ECTODERM'][column].values )
            del values_ECTODERM['0']
            values_ENDODERM = Counter(subsets_of_df['ENDODERM'][column].values)
            del values_ENDODERM['0']

            values_NEURAL_CREST = Counter(subsets_of_df['NEURAL_CREST'][column].values)
            del values_NEURAL_CREST['0']

            values_MESODERM = Counter(subsets_of_df['MESODERM'][column].values)
            del values_MESODERM['0']

            for chromosome in chromosomes:
                if chromosome not in values_ECTODERM.keys():
                    values_ECTODERM[chromosome] = 0
                if chromosome not in values_ENDODERM.keys():
                    values_ENDODERM[chromosome] = 0
                if chromosome not in values_NEURAL_CREST.keys():
                    values_NEURAL_CREST[chromosome] = 0
                if chromosome not in values_MESODERM.keys():
                    values_MESODERM[chromosome] = 0

            plt.bar(np.array(values_ECTODERM.keys()),np.array(values_ECTODERM.values()), label='ECTODERM')
            plt.bar(np.array(values_ENDODERM.keys()),np.array(values_ENDODERM.values()), label='ENDODERM', bottom=np.array(values_ECTODERM.values()))
            plt.bar(np.array(values_NEURAL_CREST.keys()),np.array(values_NEURAL_CREST.values()), label='NEURAL_CREST',bottom=np.array(values_ECTODERM.values()) + np.array(values_ENDODERM.values()))
            plt.bar(np.array(values_MESODERM.keys()),np.array(values_MESODERM.values()), label='MESODERM',bottom=np.array(values_ECTODERM.values()) + np.array(values_ENDODERM.values()) + np.array(values_NEURAL_CREST.values()))

        else:
            values_ECTODERM= subsets_of_df['ECTODERM'][column].values
            values_ECTODERM= values_ECTODERM[values_ECTODERM!= 0]

            values_ENDODERM = subsets_of_df['ENDODERM'][column].values
            values_ENDODERM = values_ENDODERM[values_ENDODERM != 0]

            values_NEURAL_CREST = subsets_of_df['NEURAL_CREST'][column].values
            values_NEURAL_CREST = values_NEURAL_CREST[values_NEURAL_CREST != 0]

            values_MESODERM = subsets_of_df['MESODERM'][column].values
            values_MESODERM = values_MESODERM[values_MESODERM != 0]

            plt.hist([values_ECTODERM, values_ENDODERM, values_MESODERM, values_NEURAL_CREST], bins = 25,stacked=True,label=['ECTO','ENDO','MESO','NEURAL'])

        plt.title(column)
        plt.legend()
        plt.savefig('../../data_chromosome/plots/tra_graphs_stats/' + column + '.png')
        plt.close()



def main_gspan_original_graphs():
    # This variables are global to simplify testing the code, will be removed later:
    NUMBER_OF_SAMPLES = -1

    MIN_GRAPHS = 100

    MAX_DISTANCE = 2000

    supports = [1, ]  # 0.9, 0.8]

    time_per_support = {}
    for support in supports:
        init = time.time()

        MIN_SUPPORT = support
        # Data paths:
        DATAPATH = '../../data_chromosome'

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
        #                                      gspan_data_folder=GSPAN_DATA_FOLDER, data_path=DATAPATH,
        #                                      processed_path=PROCESSED_PATH)
        file_path = '../../data_chromosome/raw_processed_data/data_2597_1_2000.pkl'
        name = 'classification_dataset_' + str(NUMBER_OF_SAMPLES) + '_' + str(MIN_SUPPORT) + '_' + str(
            MAX_DISTANCE) + '_nan'

        generate_dataset_gspan(path=file_path, data_path=DATAPATH, min_graphs=MIN_GRAPHS, name=name)

        end_time = timedelta(seconds=time.time() - init)
        time_per_support[support] = end_time

    for key in time_per_support.keys():
        print 'support:', key
        print 'time:', time_per_support[key]

def main_generate_gpsan_data():
    NUMBER_OF_SAMPLES = -1
    # Data paths:
    DATA_PATH = '../../data_chromosome'
    OUTPUT_PATH = DATA_PATH + '/datasets'
    GSPAN_PATH = DATA_PATH + '/new_gspan_reps/'
    MIN_SUPPORT = 1
    MIN_GRAPHS = 5

    data_path = DATA_PATH + '/raw_original_data/allfiles'
    all_patients = os.listdir(data_path)[:NUMBER_OF_SAMPLES]
    all_patients = [p.replace('.vcf.tsv','') for p in all_patients]
    print('Generating the raw data...')

    file_path = process_list_of_patients(all_patients, min_support=MIN_SUPPORT,
                                         gspan_data_folder=GSPAN_PATH, data_path=DATA_PATH,
                                         output_path=OUTPUT_PATH)
    name = 'classification_dataset_' + str(NUMBER_OF_SAMPLES) + '_' + 'only_graphs'

    generate_dataset_gspan(path=file_path, data_path=DATA_PATH, min_graphs=MIN_GRAPHS, name=name)

def main_generate_only_tra_graphs_dataset():
    NUMBER_OF_SAMPLES = -1
    # Data paths:
    DATA_PATH = '../../data_chromosome'

    data_path = DATA_PATH + '/raw_original_data/allfiles'
    all_patients = os.listdir(data_path)[:NUMBER_OF_SAMPLES]
    all_patients = [p.replace('.vcf.tsv', '') for p in all_patients]
    print('Generating the raw data...')
    generate_tra_graph_related_dataset(all_patients, DATA_PATH, name='data_only')


def main():
    NUMBER_OF_SAMPLES = 5
    # Data paths:
    DATA_PATH = '../../data_chromosome'
    OUTPUT_PATH = DATA_PATH + '/datasets'
    GSPAN_PATH = DATA_PATH + '/new_gspan_reps/'

    try: os.mkdir(DATA_PATH)
    except: pass
    try: os.mkdir(OUTPUT_PATH)
    except: pass
    try: os.mkdir(GSPAN_PATH)
    except: pass

    # Directory containing the files
    data_path = DATA_PATH + '/raw_original_data/allfiles'
    all_patients = os.listdir(data_path)
    all_patients = [p.replace('.vcf.tsv','') for p in all_patients]
    for patient in all_patients[:NUMBER_OF_SAMPLES]:
        print patient
        filename = patient
        get_traslocation_graph(filename, DATA_PATH, gspan_path=GSPAN_PATH)


if __name__ == '__main__':
    init = time.time()
    # main_generate_gpsan_data()
    # main_generate_only_tra_graphs_dataset()
    plot_graph_info('../../data_chromosome/datasets_only_graphs/data_only_tra_graphs.csv')
    print'Total time:', timedelta(seconds=time.time() - init)
