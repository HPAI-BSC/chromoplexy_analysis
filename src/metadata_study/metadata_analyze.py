"""
TODO:
    find a good strategy for imputing missing data,
        for now i'm ussing the most frequent value
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

DATAPATH = '../../data/metadatos_v2.0.txt'


def generate_csv():
    """
    This function generates a real dataset using the txt given and saves it as a csv.
    :return:
    """
    data = pd.DataFrame(
        columns=['sampleID', 'donor_sex', 'donor_age_at_diagnosis', 'histology_tier1', 'histology_tier2',
                 'tumor_stage1', 'tumor_stage2'])

    with open(DATAPATH) as f:
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

    data.to_csv('../../data/raw_original_data//metadatos_v2.0.csv', index=False)


def preprocessing(df):
    """
    Courrent strategy for nans: replace with the most_frequent
    TODO: replace with the most common age per cancer type.

    Droped tumor_stage1 and tumor_stage2
    :param df:
    :return:
    """
    # Imput missing data with the most frequent
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(df['donor_age_at_diagnosis'].values.reshape(-1, 1))
    df['donor_age_at_diagnosis'] = imp.fit_transform(df[['donor_age_at_diagnosis']]).ravel()
    df['donor_age_at_diagnosis'] = df['donor_age_at_diagnosis'].astype(np.int)

    # Drop tumor stage 1, it's very unbalanced
    df = df.drop('tumor_stage1', axis=1)
    # Drop tumor stage 2, it's very unbalanced + useless
    df = df.drop('tumor_stage2', axis=1)

    df.to_csv('../../data/raw_original_data/clean_metadata.csv', index=False)

    return df


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
    # data = pd.read_csv('../../data/metadatos_v2.0.csv')
    # data = preprocessing(data)
    clean = pd.read_csv('../../data/datasets/classification_dataset_-1_0.7_2000.csv')
    describe(clean)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
