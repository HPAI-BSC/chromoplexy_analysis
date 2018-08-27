import sys
import os

sys.path.insert(1, '../../src')

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from datetime import timedelta
import time
import pandas as pd

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn import metrics
import numpy as np

from scipy import stats

DATAPATH = '../../data'



def compare_dummy_classifiers(X_train, y_train, X_test, y_test):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Naive Bayes", "LDA"]
    classifiers = [
        KNeighborsClassifier(4),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        GaussianNB(),
        LinearDiscriminantAnalysis()]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name, score)


def compare_complex_classifiers(X_train, y_train, X_test, y_test,name):
    n_iter_search = 70
    f= open('../../data/best_params'+name+'.txt','w')
    # RDA
    parameter_distributions = {'reg_param': stats.uniform(0, 1)}
    rda = QuadraticDiscriminantAnalysis(priors=2)
    random_search = RandomizedSearchCV(rda, param_distributions=parameter_distributions, n_iter=n_iter_search,
                                       pre_dispatch=2, n_jobs=-1)
    random_search.fit(X_train, y_train.values.ravel())
    score = random_search.score(X_test, y_test)
    print 'RDA', score

    # Perceptron
    param_dist = {"penalty": [None, 'l2', 'l1', 'elasticnet'],
                  "alpha": stats.uniform(0.001, 0.05),
                  "fit_intercept": [True, False]
                  }

    per = Perceptron(n_jobs=-1, warm_start=True)

    random_search = RandomizedSearchCV(per, param_distributions=param_dist,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)
    random_search.fit(X_train, y_train.values.ravel())
    f.write(str(random_search.best_estimator_))
    f.write('\n')

    score = random_search.score(X_test, y_test)
    f.write(str(score))
    print 'Perceptron', score

    # MLP
    param_dist = {'learning_rate': ['constant','invscaling','adaptive'],
                    'alpha':stats.uniform(0.0001, 0.05),
                    'hidden_layer_sizes': stats.randint(4, 12),
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    }
    mlp = MLPClassifier(solver='adam',warm_start=True)
    random_search = RandomizedSearchCV(mlp, param_distributions=param_dist,
                                       n_iter=n_iter_search,pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())

    f.write(str(random_search.best_estimator_))
    f.write('\n')
    score = random_search.score(X_test, y_test)
    f.write(str(score))
    print 'MLP', score

    # Random forest

    param_dist = {"max_depth": [3, None],
                  "max_features": stats.randint(1, 11),
                  "min_samples_split": stats.randint(2, 11),
                  "min_samples_leaf": stats.randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    clf = RandomForestClassifier(n_estimators=20)

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())

    f.write(str(random_search.best_estimator_))
    f.write('\n')

    score = random_search.score(X_test, y_test)
    f.write(str(score))
    print 'Random Forest', score

    # Adaboost

    param_dist = {'learning_rate':stats.uniform(0.01, 1)}

    clf = AdaBoostClassifier(n_estimators=100)

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())

    f.write(str(random_search.best_estimator_))
    score = random_search.score(X_test, y_test)
    print 'Adaboost', score



def test_with_some_datasets(dataset_files):
    for dataset_file in dataset_files:
        path = DATAPATH + '/datasets/' + dataset_file
        try:
            df = pd.read_csv(path)
            y = df.pop('histology_tier1')
            # [ 'donor_age_at_diagnosis', 'donor_sex']
            X = df.drop(['Unnamed: 0', 'histology_tier2'], axis=1)
            X[X.dtypes[(X.dtypes == "float64") | (X.dtypes == "int64")]
                .index.values].hist(figsize=[11, 11])
            # plt.show()
            X_train, X_test, Y_train, Y_test = \
                train_test_split(pd.get_dummies(X), y, test_size=.2, random_state=42)
            scaler = MinMaxScaler()
            X_train[['donor_age_at_diagnosis']] = scaler.fit_transform(X_train[['donor_age_at_diagnosis']])
            X_test[['donor_age_at_diagnosis']] = scaler.fit_transform(X_test[['donor_age_at_diagnosis']])

            for column in X_train.columns:
                if 'chr' in column:
                    X_train[[column]] = scaler.fit_transform(X_train[[column]])
                    X_test[[column]] = scaler.fit_transform(X_test[[column]])

                if 'number_of_breaks' in column:
                    X_train[[column]] = scaler.fit_transform(X_train[[column]])
                    X_test[[column]] = scaler.fit_transform(X_test[[column]])

                if 'DUP' or 'DEL' or 'TRA' or 'h2hINV' or 't2tINV' in column:
                    X_train[[column]] = scaler.fit_transform(X_train[[column]])
                    X_test[[column]] = scaler.fit_transform(X_test[[column]])
            print 'Dataset', dataset_file
            # print 'Columns:', X.columns
            compare_complex_classifiers(X_train, Y_train, X_test, Y_test,name=dataset_file)
        except Exception as e:
            print(e)
            # print(dataset_file, 'does not exist')


def main():
    # datasets = ['classification_dataset_2601_0.8_2000.csv', 'classification_dataset_2601_0.8_1500.csv']
    datasets = os.listdir(DATAPATH + '/datasets/')
    test_with_some_datasets(datasets)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
