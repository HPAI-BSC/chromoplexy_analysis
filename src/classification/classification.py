import sys

sys.path.insert(1, '../../src')

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from datetime import timedelta
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
DATAPATH = '../../data'


def compare_classifiers(X_train,y_train,X_test,y_test):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(4),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name, score)


def main():
    # path = DATAPATH + '/classification_dataset_1000_0.8_1500.csv'
    path = DATAPATH + '/classification_2601_0.8_2000.csv'
    df = pd.read_csv(path)
    y = df.pop('histology_tier1')
    X = df.drop(['Unnamed: 0','histology_tier2', 'donor_age_at_diagnosis', 'donor_sex'], axis=1)
    X[X.dtypes[(X.dtypes == "float64") | (X.dtypes == "int64")]
        .index.values].hist(figsize=[11, 11])
    # plt.show()
    X_train, X_test, Y_train, Y_test = \
        train_test_split(pd.get_dummies(X), y, test_size=.2, random_state=42)
    scaler = MinMaxScaler()
    # X_train[['donor_age_at_diagnosis']] = scaler.fit_transform(X_train[['donor_age_at_diagnosis']])
    # X_test[['donor_age_at_diagnosis']] = scaler.fit_transform(X_test[['donor_age_at_diagnosis']])

    compare_classifiers(X_train,Y_train,X_test,Y_test)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
