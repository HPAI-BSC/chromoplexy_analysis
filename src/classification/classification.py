"""
TODO: Try nan imputation knn
TODO: Order this code.
TODO: Tree exploration.
"""
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
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
import numpy as np

from scipy import stats

DATAPATH = '../../data_chromosome'


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


def compare_complex_classifiers(X_train, y_train, X_test, y_test, name):
    n_iter_search = 70
    f = open('../../data/best_params' + name + '.txt', 'w')
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
    param_dist = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                  'alpha': stats.uniform(0.0001, 0.05),
                  'hidden_layer_sizes': stats.randint(4, 12),
                  'activation': ['identity', 'logistic', 'tanh', 'relu'],
                  }
    mlp = MLPClassifier(solver='adam', warm_start=True)
    random_search = RandomizedSearchCV(mlp, param_distributions=param_dist,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

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

    param_dist = {'learning_rate': stats.uniform(0.01, 1)}

    clf = AdaBoostClassifier(n_estimators=100)

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())

    f.write(str(random_search.best_estimator_))
    score = random_search.score(X_test, y_test)
    print 'Adaboost', score


import itertools
import numpy as np


import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, name='patata'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    plt.figure(figsize=(10, 12))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plot_path = DATAPATH + '/plots/confusion_matrix/'
    plt.savefig(plot_path + title + name + '.png')


def only_random_forest(X_train, y_train, X_test, y_test, name):
    # Random forest
    n_iter_search = 30
    # f = open('../../data/best_params' + 'random_forest' + name + '.txt', 'w')
    param_dist = {"max_depth": stats.randint(1, 11),
                  "min_samples_split": stats.randint(2, 11),
                  "min_samples_leaf": stats.randint(1, 11),
                  "bootstrap": [True, False],
                  # "oob_score": [True, False],
                  "criterion": ["gini", "entropy"]}

    clf = RandomForestClassifier(n_estimators=30)

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())

    best_params = random_search.best_params_
    best_params['n_estimators'] = 30

    random_forest = RandomForestClassifier(**best_params)

    random_forest = random_forest.fit(X_train, y_train)

    from sklearn import tree
    import pydotplus

    i = 0
    for tree_in_forest in random_forest.estimators_[:1]:
        # Create DOT data
        dot_data = tree.export_graphviz(tree_in_forest, out_file=None,
                                        feature_names=list(X_train.columns),
                                        class_names = random_forest.classes_
                                        )

        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)

        # Show graph
        graph.create_png()
        graph.write_png(name + str(i) + ".png")
        i+=1

    feature_importances = pd.DataFrame(random_forest.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    plot_feature_importance(feature_importances, name)

    # f.write(str(random_search.best_estimator_))
    # f.write('\n')
    score = random_forest.score(X_test, y_test)

    # f.write('Random Forest ' + str(score))
    print 'Random Forest', score
    y_test_pred = random_search.predict(X_test)
    # Compute confusion matrix
    class_names = ['ECTODERM', 'NEURAL_CREST', 'MESODERM', 'ENDODERM']
    cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred,
                                          labels=class_names)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization', name=name)

    # # Plot normalized confusion matrix
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix', name=name)



def plot_feature_importance(feature_importance, name):
    plt.clf()
    plt.figure(figsize=(20, 10))
    values0 = feature_importance.loc[feature_importance['importance'] > 0].index
    values1 = feature_importance.loc[feature_importance['importance'] > 0]['importance'].values

    plt.bar(values0, values1)
    plt.xticks(values0, rotation=90)
    plt.title('Feature importance')
    plot_path = DATAPATH + '/plots/'
    # plt.show()
    try:
        plt.savefig(plot_path + '/feature_importance/' + 'feature_importance_' + name + '.png')
    except:
        os.mkdir(plot_path + '/feature_importance/')


def nan_imputing(df):
    """
    Courrent strategy for nans: MICE
    :param df:
    :return:
    """
    # Imput missing data with mice
    from fancyimpute import MICE
    fancy_imputed = df
    dummies = pd.get_dummies(df)
    imputed = pd.DataFrame(data=MICE(verbose=False).complete(dummies), columns=dummies.columns, index=dummies.index)
    fancy_imputed.donor_age_at_diagnosis = imputed.donor_age_at_diagnosis
    fancy_imputed['donor_age_at_diagnosis'] = fancy_imputed['donor_age_at_diagnosis'].astype(np.int)
    return fancy_imputed

def test_with_some_datasets_only_graphs(dataset_files):
    for dataset_file in dataset_files:
        path = DATAPATH + '/datasets/' + dataset_file
        # try:
        if '.csv' in dataset_file and 'graph' not in dataset_file:
            try:
                df = pd.read_csv(path)
                y = df.pop('histology_tier1')

                # [ 'donor_age_at_diagnosis', 'donor_sex']
                X = df.drop(['Unnamed: 0', 'histology_tier2','donor_age_at_diagnosis','donor_sex','tumor_stage1','tumor_stage2','number_of_breaks'], axis=1)
                for column in X.columns:
                    if 'chr' in column:
                        X = X.drop(column,axis=1)
                    if 'DUP' in column:
                        X = X.drop(column,axis=1)
                    if 'DEL' in column:
                        X = X.drop(column, axis=1)
                    if 'TRA' in column:
                        X = X.drop(column, axis=1)
                    if 'h2hINV' in column:
                        X = X.drop(column, axis=1)
                    if 't2tINV' in column:
                        X = X.drop(column, axis=1)

                X_train, X_test, Y_train, Y_test = \
                    train_test_split(pd.get_dummies(X), y,stratify=y, test_size=.2, random_state=42)
                print 'Dataset', dataset_file
                only_random_forest(X_train, Y_train, X_test, Y_test, name=dataset_file)
                X_train['histology_tier1'] = Y_train
                X_test['histology_tier1'] = Y_test
                X_train.to_csv(DATAPATH + '/datasets/clean/' + dataset_file + '_clean.csv')
            except:
                pass


def test_with_some_datasets(dataset_files):
    for dataset_file in dataset_files:
        path = DATAPATH + '/datasets/' + dataset_file
        try:
            if '.csv' in dataset_file and 'graph' not in dataset_file:
                df = pd.read_csv(path)
                y = df.pop('histology_tier1')
                X = df.drop(['Unnamed: 0', 'histology_tier2'], axis=1)
                X_train, X_test, Y_train, Y_test = \
                    train_test_split(pd.get_dummies(X), y,stratify=y, test_size=.2, random_state=42)
                X_train = nan_imputing(X_train)
                X_test = nan_imputing(X_test)
                X_train['number_of_breaks'] = X_train['DUP'] + X_train['DEL'] + X_train['TRA'] + X_train['h2hINV'] + X_train['t2tINV']
                X_test['number_of_breaks'] = X_test['DUP'] + X_test['DEL'] + X_test['TRA'] + X_test['h2hINV'] + X_test['t2tINV']
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

                print 'Dataset', dataset_file
                only_random_forest(X_train, Y_train, X_test, Y_test, name=dataset_file+'_meta')

                X_train['histology_tier1'] = Y_train
                X_test['histology_tier1'] = Y_test
                X_train.to_csv(DATAPATH + '/datasets/clean/' + dataset_file + '_clean_meta.csv')
        except Exception as e:
            print(dataset_file)
            print(e)
            print(dataset_file, 'does not exist')


def main():
    # datasets = ['classification_dataset_2601_0.8_2000.csv', 'classification_dataset_2601_0.8_1500.csv']
    datasets = os.listdir(DATAPATH + '/datasets/')
    test_with_some_datasets_only_graphs(datasets)
    test_with_some_datasets(datasets)


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)
