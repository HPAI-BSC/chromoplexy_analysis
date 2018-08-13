"""
TODO:
    find a good strategy for imputing missing data,
        for now i'm ussing the most frequent value

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import time
from datetime import timedelta

import warnings
warnings.simplefilter('ignore')

import seaborn as sns
sns.set()

sys.path.insert(1, '../../src')

DATAPATH = '../../data/metadatos_v2.0.txt'


def generate_csv():
    data = pd.DataFrame(
        columns=['sampleID', 'donor_sex', 'donor_age_at_diagnosis', 'histology_tier1', 'histology_tier2',
                 'tumor_stage1','tumor_stage2'])

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
            data = data.append({'sampleID':id, 'donor_sex':sex, 'donor_age_at_diagnosis':age,
                                'histology_tier1':tier1, 'histology_tier2':tier2,
                 'tumor_stage1':tumor_stage1,'tumor_stage2':tumor_stage2},ignore_index=True)

    data = data.drop(data.index[0])

    data.to_csv('../../data/metadatos_v2.0.csv',index=False)

def nan_processing(df):
    data_with_na = df[df.isnull().any(axis=1)]
    print df['histology_tier1'].value_counts()
    print data_with_na['histology_tier1'].value_counts()
    print df['histology_tier2'].value_counts()
    print data_with_na['histology_tier2'].value_counts()
    print df['histology_tier1'].value_counts()
    print data_with_na['histology_tier1'].value_counts()
    print df['histology_tier2'].value_counts()
    print data_with_na['histology_tier2'].value_counts()
    print df['tumor_stage1'].value_counts()
    print data_with_na['tumor_stage1'].value_counts()
    print df['tumor_stage2'].value_counts()
    print data_with_na['tumor_stage2'].value_counts()


def describe(df):
    print df.head()
    print df.shape
    # print df.describe(include=['float64', 'object'])
    print('age nans',     df.isnull().sum())
    df['donor_age_at_diagnosis'].hist(bins=50)

    # needs testing
    for col in df.columns:
        print df[col].value_counts()
        if col ==
        plt.figure()
        sns.countplot(x=col,data=df)
        plt.show()



def main():
    # generate_csv()
    data = pd.read_csv('../../data/metadatos_v2.0.csv')
    # nan_processing(data)
    describe(data)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
