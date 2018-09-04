"""
TODO:
    find a good strategy for imputing missing data,
        for now i'm ussing the most frequent value

        Bone cancer imputer: https://seer.cancer.gov/statfacts/html/bones.html

        Breast cancer age imputer:https://seer.cancer.gov/statfacts/html/breast.html
            Distribution

    If the classes doesen't work I can merge ECTODERM and NEURAL_CREST
"""
import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import time
from datetime import timedelta
from collections import Counter, OrderedDict
import os
import seaborn as sns

sns.set()

from sklearn.preprocessing import Imputer

sys.path.insert(1, '../../src')

DATAPATH = '../../data_chromosome/'

METADATAPATH = DATAPATH + 'raw_original_data/data/metadatos_v2.0.txt'


try:
    os.mkdir(DATAPATH + 'plots')
except:
    pass

try:
    os.mkdir(DATAPATH + 'plots/tra_study')
except:
    pass

try:
    os.mkdir(DATAPATH + 'plots/data_analyisis')
except:
    pass

def generate_csv():
    """
    This function generates a real dataset using the txt given and saves it as a csv.
    :return:
    """
    data = pd.DataFrame(
        columns=['sampleID', 'donor_sex', 'donor_age_at_diagnosis', 'histology_tier1', 'histology_tier2',
                 'tumor_stage1', 'tumor_stage2'])

    with open(METADATAPATH) as f:
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

    data.to_csv('../../data/raw_original_data/metadatos_v2.0.csv', index=False)


def study_donor_age_vs_histology():
    df = pd.read_csv('../../data/raw_original_data/metadatos_v2.0.csv')
    df_nans = df[df.isnull().any(axis=1)]

    plt.figure(figsize=(15, 10))
    values = df_nans['histology_tier1'].values
    d = Counter(values)
    factor = 1.0 / sum(d.itervalues())
    D = {k: v * factor for k, v in d.iteritems()}
    plt.bar(range(len(D)), list(D.values()))
    plt.xticks(range(len(D)), list(D.keys()), rotation=30)
    plt.title('Values with nan per histology')

    plt.savefig('../../data/plots/data_analysis/values_with_nan_per_histology.png')

    df.dropna(inplace=True)
    df.donor_age_at_diagnosis = df.donor_age_at_diagnosis.astype(int)
    lower, higher = df['donor_age_at_diagnosis'].min(), df['donor_age_at_diagnosis'].max()
    n_bins = 20
    edges = range(lower, higher, (higher - lower) / n_bins)  # the number of edges is 8
    lbs = ['(%d, %d]' % (edges[i], edges[i + 1]) for i in range(len(edges) - 2)]
    values = pd.cut(df.donor_age_at_diagnosis, bins=n_bins + 1, labels=lbs, include_lowest=True)
    cross_tab = pd.crosstab(values, df.histology_tier1)
    print cross_tab
    # now stack and reset
    stacked = cross_tab.stack().reset_index().rename(columns={0: 'value'})

    # plot grouped bar chart
    plt.figure(figsize=(20, 10))
    sns.barplot(x=stacked.donor_age_at_diagnosis, y=stacked.value, hue=stacked.histology_tier1)
    plt.xticks(rotation=30)
    plt.title('Donor age vs Histology')
    plt.savefig(DATAPATH + '/plots/data_analysis/donor_age_vs_histology_without_nans.png')


from natsort import natsorted

def plot_heatmap_traslocations(all_only_tra, name):
    import seaborn as sns
    sns.set()
    plt.figure(figsize=(15, 15))

    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                   '19', '20', '21', '22', 'X', 'Y']

    all_only_tra = all_only_tra.reindex(index=natsorted(all_only_tra.index))
    all_only_tra = all_only_tra[chromosomes]
    all_only_tra = all_only_tra.astype(int)
    all_only_tra = pd.DataFrame(data=np.maximum(all_only_tra.values, all_only_tra.values.transpose()),
                                columns=chromosomes, index=chromosomes, )

    print(all_only_tra)
    g = sns.heatmap(all_only_tra, annot=True, linewidths=.5, cbar=False, fmt='g')
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    plt.title('Traslocations per chromosome of ' + name)
    plt.savefig(DATAPATH + 'plots/tra_study/' + 'traslocations_' + name + '.png')


def deletion_frequency_per_chromosome():
    NUMBER_OF_SAMPLES = -1

    # Directory containing the files
    data_path = DATAPATH + '/raw_original_data/allfiles/'
    all_patients = os.listdir(data_path)[:NUMBER_OF_SAMPLES]

    df = pd.read_csv('../../data/raw_original_data/metadatos_v2.0.csv')

    df = df.set_index('sampleID')

    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', 'X', 'Y']

    all_ct = pd.DataFrame(columns=['DEL', 'DUP', 'TRA', 'h2hINV', 't2tINV'],
                          index=chromosomes)

    all_ct_ECTODERM = pd.DataFrame(columns=['DEL', 'DUP', 'TRA', 'h2hINV', 't2tINV'],
                                   index=chromosomes)
    all_ct_ENDODERM = pd.DataFrame(columns=['DEL', 'DUP', 'TRA', 'h2hINV', 't2tINV'],
                                   index=chromosomes)
    all_ct_NEURAL_CREST = pd.DataFrame(columns=['DEL', 'DUP', 'TRA', 'h2hINV', 't2tINV'],
                                       index=chromosomes)
    all_ct_MESODERM = pd.DataFrame(columns=['DEL', 'DUP', 'TRA', 'h2hINV', 't2tINV'],
                                   index=chromosomes)

    all_only_tra = pd.DataFrame(0,columns=chromosomes, index=chromosomes,)

    all_tra_ECTODERM = pd.DataFrame(0,columns=chromosomes, index=chromosomes,)
    all_tra_ENDODERM = pd.DataFrame(0,columns=chromosomes, index=chromosomes,)
    all_tra_NEURAL_CREST = pd.DataFrame(0,columns=chromosomes, index=chromosomes,)
    all_tra_MESODERM = pd.DataFrame(0,columns=chromosomes, index=chromosomes,)



    all_ct.index = all_ct.index.map(str)

    all_ct_ECTODERM.index = all_ct_ECTODERM.index.map(str)
    all_ct_ENDODERM.index = all_ct_ENDODERM.index.map(str)
    all_ct_NEURAL_CREST.index = all_ct_NEURAL_CREST.index.map(str)
    all_ct_MESODERM.index = all_ct_MESODERM.index.map(str)

    all_only_tra.index = all_only_tra.index.map(str)
    all_tra_ECTODERM.index = all_tra_ECTODERM.index.map(str)
    all_tra_ENDODERM.index = all_tra_ENDODERM.index.map(str)
    all_tra_NEURAL_CREST.index = all_tra_NEURAL_CREST.index.map(str)
    all_tra_MESODERM.index = all_tra_MESODERM.index.map(str)


    all_ct = all_ct.reindex(index=natsorted(all_ct.index))

    for patient in all_patients:

        patient_path = data_path + patient
        # load patient breaks
        patient_breaks = pd.DataFrame.from_csv(patient_path, sep='\t', index_col=None)
        # load the chromosomes as strings
        patient_breaks['chrom2'] = patient_breaks['chrom2'].map(str)
        # generate a crosstab of the svclass with the chromosomes
        ct = pd.crosstab(patient_breaks['chrom2'], patient_breaks['svclass'])
        ct.index = ct.index.map(str)

        all_ct = all_ct.add(ct, fill_value=0)
        patient = patient.replace('.vcf.tsv','')
        # Ignore the patients witout metadata
        try:
            if df.loc[patient, 'histology_tier1'] == 'ECTODERM':
                all_ct_ECTODERM = all_ct_ECTODERM.add(ct, fill_value=0)
            if df.loc[patient, 'histology_tier1'] == 'ENDODERM':
                all_ct_ENDODERM = all_ct_ENDODERM.add(ct, fill_value=0)
            if df.loc[patient, 'histology_tier1'] == 'NEURAL_CREST':
                all_ct_NEURAL_CREST = all_ct_NEURAL_CREST.add(ct, fill_value=0)
            if df.loc[patient, 'histology_tier1'] == 'MESODERM':
                all_ct_MESODERM = all_ct_MESODERM.add(ct, fill_value=0)
        except:
            pass
        # The TRA classes have different chromosome1 and chromosome2, I want to make a heatmap of the changes.

        only_TRA = patient_breaks.loc[patient_breaks['svclass'] == 'TRA']
        ct_tra = pd.crosstab(only_TRA['#chrom1'], only_TRA['chrom2'])
        ct_tra.index = ct_tra.index.map(str)
        all_only_tra = all_only_tra.add(ct_tra, fill_value=0)
        try:
            if df.loc[patient, 'histology_tier1'] == 'ECTODERM':
                all_tra_ECTODERM = all_tra_ECTODERM.add(ct_tra, fill_value=0)
            if df.loc[patient, 'histology_tier1'] == 'ENDODERM':
                all_tra_ENDODERM = all_tra_ENDODERM.add(ct_tra, fill_value=0)
            if df.loc[patient, 'histology_tier1'] == 'NEURAL_CREST':
                all_tra_NEURAL_CREST = all_tra_NEURAL_CREST.add(ct_tra, fill_value=0)
            if df.loc[patient, 'histology_tier1'] == 'MESODERM':
                all_tra_MESODERM = all_tra_MESODERM.add(ct_tra, fill_value=0)
        except:
            pass

    plot_heatmap_traslocations(all_only_tra, 'all')

    plot_heatmap_traslocations(all_tra_ECTODERM, 'ECTODERM')
    plot_heatmap_traslocations(all_tra_ENDODERM, 'ENDODERM')
    plot_heatmap_traslocations(all_tra_NEURAL_CREST, 'NEURAL_CREST')
    plot_heatmap_traslocations(all_tra_MESODERM, 'MESODERM')

    # plt.figure(figsize=(20, 10))
    #
    # values = all_ct['DEL'].values
    #
    # all_ct = all_ct.reindex(index=natsorted(all_ct.index))
    # plt.bar(all_ct.index, all_ct['TRA'])
    #
    # # all_ct_ECTODERM = all_ct_ECTODERM.reindex(index=natsorted(all_ct_ECTODERM.index)).fillna(0)
    # # all_ct_ENDODERM = all_ct_ENDODERM.reindex(index=natsorted(all_ct_ENDODERM.index)).fillna(0)
    # # all_ct_NEURAL_CREST = all_ct_NEURAL_CREST.reindex(index=natsorted(all_ct_NEURAL_CREST.index)).fillna(0)
    # # all_ct_MESODERM = all_ct_MESODERM.reindex(index=natsorted(all_ct_MESODERM.index)).fillna(0)
    # # print all_ct
    # # ax = plt.subplot()
    # # # plt.bar(all_ct.index, all_ct['DEL'])
    # # ax.bar(all_ct_ECTODERM.index, all_ct_ECTODERM['DEL'],label='ECTODERM')
    # # ax.bar(all_ct_ENDODERM.index, all_ct_ENDODERM['DEL'], label='ENDODERM')
    # # ax.bar(all_ct_NEURAL_CREST.index, all_ct_NEURAL_CREST['DEL'],label='NEURAL_CREST')
    # # ax.bar(all_ct_MESODERM.index, all_ct_MESODERM['DEL'], label='MESODERM')
    # # print(all_ct_ECTODERM)
    # # plt.xticks(range(len(D)), list(D.keys()), rotation=30)
    # plt.title('Frecuency of Deletions')
    # plt.legend()
    # plt.show()


def nan_imputing(df):
    """
    Courrent strategy for nans: replace with the most_frequent
    TODO: replace with the most common age per cancer type.

    Droped tumor_stage1 and tumor_stage2
    :param df:
    :return:
    """
    # Imput missing data with knn
    from fancyimpute import MICE, KNN
    fancy_imputed = df
    print(df.head())
    dummies = pd.get_dummies(df.drop('sampleID', axis=1))
    imputed = pd.DataFrame(data=MICE().complete(dummies), columns=dummies.columns, index=dummies.index)
    fancy_imputed.donor_age_at_diagnosis = imputed.donor_age_at_diagnosis

    # imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    # imp.fit(df['donor_age_at_diagnosis'].values.reshape(-1, 1))
    # df['donor_age_at_diagnosis'] = imp.fit_transform(df[['donor_age_at_diagnosis']]).ravel()
    fancy_imputed['donor_age_at_diagnosis'] = fancy_imputed['donor_age_at_diagnosis'].astype(np.int)

    # Drop tumor stage 1, it's very unbalanced
    # df = df.drop('tumor_stage1', axis=1)
    # # Drop tumor stage 2, it's very unbalanced + useless
    # df = df.drop('tumor_stage2', axis=1)

    fancy_imputed.to_csv('../../data/raw_original_data/clean_metadata_mice.csv', index=False)

    return fancy_imputed


def describe(df):
    """
    This function prints a report of the metadata and the representative plots
    :param df:
    :return:
    """
    print 'Dataset:'
    print df.head()
    print 'Shape:'
    print df.shape
    print 'Crosstab between histology tier1 and histology tier2: '
    print pd.crosstab(df.histology_tier1, df.histology_tier2)
    for col in df.columns:
        # don't print sample ID
        if col == 'sampleID':
            continue

        plt.figure(figsize=(20, 10))

        values = df[col].values
        if col == 'donor_age_at_diagnosis':
            lower, higher = df['donor_age_at_diagnosis'].min(), df['donor_age_at_diagnosis'].max()
            n_bins = 20
            edges = range(lower, higher, (higher - lower) / n_bins)  # the number of edges is 8
            lbs = ['(%d, %d]' % (edges[i], edges[i + 1]) for i in range(len(edges) - 2)]
            values = pd.cut(df.donor_age_at_diagnosis, bins=n_bins + 1, labels=lbs, include_lowest=True)

        d = Counter(values)
        factor = 1.0 / sum(d.itervalues())
        D = {k: v * factor for k, v in d.iteritems()}

        if col == 'donor_age_at_diagnosis':
            D = OrderedDict(
                (k, v) for k, v in sorted(D.iteritems(), key=lambda (k, v): (int(k[1:-1].split(',')[0]), v)))

        plt.bar(range(len(D)), list(D.values()))
        plt.xticks(range(len(D)), list(D.keys()), rotation=30)
        plt.title(col)
        plot_path = '../../data/plots/data_analysis/'
        try:
            os.mkdir(plot_path)
        except:
            pass

        plt.savefig(plot_path + 'barplot_' + col)


def main():
    # generate_csv()
    # data = pd.read_csv('../../data/raw_original_data/metadatos_v2.0.csv')
    # clean = nan_imputing(data)
    # clean = pd.read_csv('../../data/datasets/classification_dataset_-1_0.7_2000.csv')
    deletion_frequency_per_chromosome()
    # describe(clean)
    # study_donor_age_vs_histology()


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
