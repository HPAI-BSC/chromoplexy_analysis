#!/usr/bin/env python
# coding: utf-8

# # In out file. 
# On this file I have to put all the methods related to the data processing. 
# * Add metadata 
# * Label stuff
# * Outlier treatment !!! TODO
# * Missings 
# * Split
# 
# I'll make it in the jupyter and then I'll move it into a py file, once it is tested and working. 
# 
# ````
# Input: data_path/allfiles  + data_path/metadatos_v2.0.txt
# Output: name.csv or name_train.csv, name_train_target.csv, name_test.csv, name_test_target.csv
# 
# ````


import os
import sys

sys.path.insert(1, '../../src')

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from datetime import timedelta
import time
import numpy as np
import pandas as pd
import networkx as nx

from fancyimpute import IterativeImputer
from sklearn.model_selection import train_test_split
from natsort import natsorted
from matplotlib import pyplot as plt
import gc

# Data paths:
DATA_PATH = '../definitive_data_folder'
PATIENTS_PATH = DATA_PATH + '/allfiles'
# The prgram will try to load the csv, if the csv does not exist it will generate it ussing the txt. 
METADATA_PATH = DATA_PATH + '/metadatos_v2.0.csv'

if not os.path.exists(METADATA_PATH):
    generate_metadata_csv()

OUTPUT_PATH = DATA_PATH + '/datasets'

try:
    os.mkdir(DATA_PATH)
except:
    pass
try:
    os.mkdir(OUTPUT_PATH)
except:
    pass

# Globals
labels = ['ECTODERM', 'NEURAL_CREST', 'MESODERM', 'ENDODERM']
hist2 = np.array(['Biliary', 'Bladder', 'Bone/SoftTissue', 'Breast', 'CNS', 'Cervix',
                  'Colon/Rectum', 'Esophagus', 'Head/Neck', 'Kidney', 'Liver',
                  'Lung', 'Lymphoid', 'Myeloid', 'Ovary', 'Pancreas', 'Prostate',
                  'Skin', 'Stomach', 'Thyroid', 'Uterus'])
chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
               '19', '20', '21', '22', 'X', 'Y']
svclass = ['DEL', 'DUP', 'TRA', 'h2hINV', 't2tINV']
k = 300
TOMMY = '43dadc68-c623-11e3-bf01-24c6515278c0'


def generate_metadata_csv():
    """
    This function generates a real dataset using the txt given and saves it as a csv.
    :return:
    """
    data = pd.DataFrame(
        columns=['sampleID', 'donor_sex', 'donor_age_at_diagnosis', 'histology_tier1', 'histology_tier2',
                 'tumor_stage1', 'tumor_stage2'])

    with open(METADATA_PATH.replace('.csv', '.txt')) as f:
        for l in f:
            words = l.split()
            id = words[0]
            sex = words[1]
            age = words[2]
            tier1 = words[3]
            tier2 = words[4]
            tumor_stage1 = '_'.join(words[5:7])
            tumor_stage2 = '_'.join(words[8:])
            data = data.append({'sampleID': id, 'donor_sex': sex, 'donor_age_at_diagnosis': age,
                                'histology_tier1': tier1, 'histology_tier2': tier2,
                                'tumor_stage1': tumor_stage1, 'tumor_stage2': tumor_stage2}, ignore_index=True)

    data = data.drop(data.index[0])
    data.to_csv(METADATA_PATH, index=False)


def generateTRAGraph(patient):
    '''
    This function generates a graph per patient representing the traslocations of this patient.
    
    vertex: Chromosomes
    edge: the number of traslocations between each chromosome

    Input:
        patient(string):  The patient id.
    Output:
        graph: networkx format
        edge_list: List with the format:
                    node1 node2 weight    (edge between node1 and node2 with weight weight)
    '''
    patient_path = PATIENTS_PATH + '/' + patient + '.vcf.tsv'

    # Load the patient breaks, and select only the traslocations
    patient_breaks = pd.read_csv(patient_path, sep='\t', index_col=None)

    # patient_breaks['chrom2'] = patient_breaks['chrom2'].map(str)

    only_TRA = patient_breaks.loc[patient_breaks['svclass'] == 'TRA']

    # The crosstab is equivalent to the adjacency matrix, so we use this to calculate it
    ct_tra = pd.crosstab(only_TRA['#chrom1'], only_TRA['chrom2'])

    ct_tra.index = ct_tra.index.map(str)
    adjacency_matrix_connected_only = ct_tra

    aux = pd.DataFrame(0, columns=chromosomes, index=chromosomes)
    aux.index = aux.index.map(str)

    ct_tra = aux.add(ct_tra, fill_value=0)
    aux = None
    # Reorder
    ct_tra = ct_tra.reindex(index=natsorted(ct_tra.index))
    ct_tra = ct_tra[chromosomes]
    # change the values to int
    ct_tra = ct_tra.astype(int)

    # Generate the adjacency matrix
    adjacency_matrix = pd.DataFrame(data=ct_tra.values,
                                    columns=chromosomes, index=chromosomes)
    # print(adjacency_matrix)
    graph = nx.from_pandas_adjacency(adjacency_matrix)
    graph.to_undirected()

    # Remove isolated vertices 
    graph.remove_nodes_from(list(nx.isolates(graph)))

    edge_list = nx.generate_edgelist(graph, data=['weight'])
    return graph, edge_list


def nan_imputing(df):
    """
    There is only one feature with nans. Donor age at diagnosis. 
    We impute it using the KNN strategy
    :param df:
    :return:
    """
    # Imput missing data with mice
    fancy_imputed = df
    dummies = pd.get_dummies(df)
    imputed = pd.DataFrame(data=IterativeImputer().fit_transform(dummies), columns=dummies.columns, index=dummies.index)
    fancy_imputed.donor_age_at_diagnosis = imputed.donor_age_at_diagnosis
    fancy_imputed['donor_age_at_diagnosis'] = fancy_imputed['donor_age_at_diagnosis'].astype(np.int)
    return fancy_imputed


def preprocessing_without_split(X):
    # this function is only ment for data analysis
    X['donor_sex'] = X['donor_sex'].str.replace('female', '1')
    X['donor_sex'] = X['donor_sex'].str.replace('male', '0')

    X['female'] = pd.to_numeric(X['donor_sex'])
    X = X.drop('donor_sex', axis=1)
    # X['number_of_breaks'] = X['DUP'] + X['DEL'] + X['TRA'] + X['h2hINV'] + X['t2tINV']
    for column in X.columns:
        if 'chr' in column:
            X['proportion_' + column] = 0
            X[['proportion_' + column]] = np.true_divide(np.float32(X[[column]]),
                                                         np.float32(X[['number_of_breaks']]))

        if 'DUP' in column or 'DEL' in column or 'TRA' in column or 'h2hINV' in column or 't2tINV' in column:
            X['proportion_' + column] = 0
            X[['proportion_' + column]] = np.true_divide(np.float32(X[[column]]),
                                                         np.float32(X[['number_of_breaks']]))
    X = nan_imputing(X)
    X = pd.get_dummies(X, columns=['tumor_stage1', 'tumor_stage2'])
    return X


def preprocessing(df, hist1=True):
    if hist1:
        y = df.pop('histology_tier1')
        X = df.drop('histology_tier2', axis=1)
    else:
        y = df.pop('histology_tier2')
        X = df.drop('histology_tier1', axis=1)

    X['donor_sex'] = X['donor_sex'].str.replace('female', '1')
    X['donor_sex'] = X['donor_sex'].str.replace('male', '0')

    X['female'] = pd.to_numeric(X['donor_sex'])

    X = X.drop('donor_sex', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(pd.get_dummies(X), y, stratify=y, test_size=.2)
    X_train = nan_imputing(X_train)
    X_test = nan_imputing(X_test)
    # X_train['number_of_breaks'] = X_train['DUP'] + X_train['DEL'] + X_train['TRA'] + X_train['h2hINV'] + \
    #                              X_train['t2tINV']
    # X_test['number_of_breaks'] = X_test['DUP'] + X_test['DEL'] + X_test['TRA'] + X_test['h2hINV'] + X_test[
    #     't2tINV']
    for column in X_train.columns:
        if 'chr' in column:
            X_train['proportion_' + column] = 0
            X_train[['proportion_' + column]] = np.true_divide(np.float32(X_train[[column]]),
                                                               np.float32(X_train[['number_of_breaks']]))
            X_test['proportion_' + column] = 0
            X_test[['proportion_' + column]] = np.true_divide(np.float32(X_test[[column]]),
                                                              np.float32(X_test[['number_of_breaks']]))

        if 'DUP' in column or 'DEL' in column or 'TRA' in column or 'h2hINV' in column or 't2tINV' in column:
            X_train['proportion_' + column] = 0
            X_train[['proportion_' + column]] = np.true_divide(np.float32(X_train[[column]]),
                                                               np.float32(X_train[['number_of_breaks']]))
            X_test['proportion_' + column] = 0
            X_test[['proportion_' + column]] = np.true_divide(np.float32(X_test[[column]]),
                                                              np.float32(X_test[['number_of_breaks']]))
    return X_train, Y_train, X_test, Y_test


def generate_dataset(name, split=True, hist1=True):
    """
    slow but u only need to run it once.
    
    connected_components
    connected_components_max_size
    """
    print 'Generating csv..'
    # load the metadata 
    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata.set_index('sampleID')

    # load the patient ids and remove the ones that don't have metadata.
    patients = os.listdir(PATIENTS_PATH)
    patients = [p.replace('.vcf.tsv', '') for p in patients if p in list(metadata.index)]

    # The initial dataset is the metadata one. 
    dataset = metadata

    for i, patient in enumerate(metadata.index):
        # Generate the traslocation graph of the patient and the edge_list
        g, edge_list = generateTRAGraph(patient=patient)

        dataset.loc[patient, 'connected_components'] = len(list(nx.connected_component_subgraphs(g)))

        # add the max of the number of vertex of the connected components of the graph
        if len(list(nx.connected_component_subgraphs(g))) > 0:
            dataset.loc[patient, 'connected_components_max_size'] = np.max(
                [len(list(component.nodes())) for component in nx.connected_component_subgraphs(g)])
        else:
            dataset.loc[patient, 'connected_components_max_size'] = 0

        # add the translocations
        for edge in edge_list:
            edge = edge.split(' ')
            if edge[0] in ['X', 'Y'] and edge[1] in ['X', 'Y']:
                edge_column = '(' + 'X' + ',' + 'Y' + ')'
            elif edge[0] in ['X', 'Y']:
                edge_column = '(' + edge[1] + ',' + edge[0] + ')'
            elif edge[1] in ['X', 'Y']:
                edge_column = '(' + edge[0] + ',' + edge[1] + ')'
            elif int(edge[0]) < int(edge[1]):
                edge_column = '(' + edge[0] + ',' + edge[1] + ')'
            else:
                edge_column = '(' + edge[1] + ',' + edge[0] + ')'
            edge_weight = int(edge[2])
            dataset.loc[patient, edge_column] = edge_weight

        # now we load the breaks
        patient_path = PATIENTS_PATH + '/' + patient + '.vcf.tsv'
        patient_breaks = pd.read_csv(patient_path, sep='\t', index_col=None)

        # load the chromosomes as strings
        patient_breaks['chrom2'] = patient_breaks['chrom2'].map(str)

        # generate a crosstab of the svclass with the chromosomes and add this info to the dataset
        ct = pd.crosstab(patient_breaks['chrom2'], patient_breaks['svclass'])
        ct.index = ct.index.map(str)

        for chrom in ct.index:
            for svc in ct.columns:
                dataset.loc[patient, svc + '_' + str(chrom)] = ct.loc[chrom, svc]

        # add the number of breaks
        number_of_breaks = len(patient_breaks)
        dataset.loc[patient, 'number_of_breaks'] = number_of_breaks

        # I count how many times appears on the breaks each of the chromosomes.
        contained_chromosomes = patient_breaks[['#chrom1', 'chrom2']].apply(pd.Series.value_counts)
        contained_chromosomes = contained_chromosomes.fillna(0)
        contained_chromosomes[['#chrom1', 'chrom2']] = contained_chromosomes[['#chrom1', 'chrom2']].astype(int)
        contained_chromosomes['chromosome'] = contained_chromosomes.index
        contained_chromosomes['count'] = contained_chromosomes['#chrom1'] + contained_chromosomes['chrom2']
        # Then saves it on the chromosome feature.
        for chrom in contained_chromosomes.index:
            dataset.loc[patient, 'chr_' + str(chrom)] = contained_chromosomes.loc[chrom, 'count']

        # Counts how many breaks of each class there are on the breaks and saves it.
        count_svclass = patient_breaks[['svclass', ]].apply(pd.Series.value_counts)
        for svclass in count_svclass.index:
            dataset.loc[patient, svclass] = count_svclass.loc[svclass, 'svclass']

    # fill with zeros the false nans generated now
    dataset.loc[:, dataset.columns != 'donor_age_at_diagnosis'] = dataset.loc[:,
                                                                  dataset.columns != 'donor_age_at_diagnosis'].fillna(0)

    if split:
        X_train, Y_train, X_test, Y_test = preprocessing(dataset, hist1)
        # and save
        X_train.to_csv(OUTPUT_PATH + '/' + name + '_train.csv')
        Y_train.to_csv(OUTPUT_PATH + '/' + name + '_train_target.csv')
        X_test.to_csv(OUTPUT_PATH + '/' + name + '_test.csv')
        Y_test.to_csv(OUTPUT_PATH + '/' + name + '_test_target.csv')
        return X_train, Y_train, X_test, Y_test
    else:
        dataset = preprocessing_without_split(dataset)
        dataset.to_csv(OUTPUT_PATH + '/' + name + '.csv')
        return dataset



def load_data(name):
    try:
        X_train = pd.read_csv(OUTPUT_PATH + '/' + name + '_train.csv', index_col=0)
        Y_train = pd.read_csv(OUTPUT_PATH + '/' + name + '_train_target.csv', index_col=0,
                              names=['SampleID', 'histology'])
        X_test = pd.read_csv(OUTPUT_PATH + '/' + name + '_test.csv', index_col=0)
        Y_test = pd.read_csv(OUTPUT_PATH + '/' + name + '_test_target.csv', index_col=0,
                             names=['SampleID', 'histology'])
        print 'Loaded'
    except Exception as e:
        print 'peta', e
        return
    return X_train, Y_train, X_test, Y_test

