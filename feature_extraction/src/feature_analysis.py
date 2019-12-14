#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter, OrderedDict
import json

from in_out import load_data, transform_y_to_only_one_class

DATA_PATH = '../definitive_data_folder'

LABELS = ['ECTODERM', 'NEURAL_CREST', 'MESODERM', 'ENDODERM']


# # Hyper parameter tuning

def calculate_best_hyperparameters(X_train, Y_train, n_iter_search):
    param_dist = {"max_depth": stats.randint(2, 20),
                  "min_samples_split": stats.randint(2, 11),
                  "min_samples_leaf": stats.randint(1, 20),
                  "bootstrap": [True, False],
                  "max_features": ['auto', 'log2', None],
                  "criterion": ["gini", "entropy"]}

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, iid=False,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, Y_train.values.ravel())
    best_params = random_search.best_params_
    best_params['n_estimators'] = 100
    best_params['class_weight'] = 'balanced'
    print('The best hyperparameters are: ', best_params)
    return best_params


# The best hyperparameters: 
# {'bootstrap': True,
#  'criterion': 'entropy',
#  'max_depth': 19,
#  'max_features': None,
#  'min_samples_leaf': 3,
#  'min_samples_split': 6, 
#  'class_weight': 'balanced',
#  'n_estimators': 100}

def load_best_hyperparameters():
    hyperparameters = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 19, 'max_features': None,
                       'min_samples_leaf': 3, 'min_samples_split': 6, 'class_weight': 'balanced', 'n_estimators': 100}
    hyperparameters['n_jobs'] = -1
    return hyperparameters


# # Feature ranking

def feature_extractor(X_train, Y_train, best_hyperparameters, number_of_iterations):
    features = list(X_train.columns)
    feature_importance = {feature: 0 for feature in features}
    best_hyperparameters['n_jobs'] = -1

    for i in range(number_of_iterations):
        if i % 10 == 0:
            print i
        random_forest = RandomForestClassifier(**best_hyperparameters)
        random_forest = random_forest.fit(X_train, Y_train)
        local_fi = np.array(pd.DataFrame(random_forest.feature_importances_,
                                         index=X_train.columns,
                                         columns=['importance']).sort_values('importance', ascending=False).index)
        j = 1
        for feature in local_fi:
            feature_importance[feature] += j
            j += 1
    D = OrderedDict((k, v) for k, v in sorted(feature_importance.iteritems(), key=lambda kv: kv[1]))
    with open(DATA_PATH + '/feature_ranking.json', 'w') as f:
        f.write(json.dumps(D))
    return D


# # Feature ranking One vs All

def feature_extractor_one_vs_all(X_train, Y_train, best_hyperparameters, number_of_iterations):
    all_rankings = {}
    best_hyperparameters['n_jobs'] = -1
    for class_name in LABELS:
        print class_name
        Y_train_class = transform_y_to_only_one_class(Y_train, class_name)
        features = list(X_train.columns)
        feature_importance = {feature: 0 for feature in features}
        for i in range(number_of_iterations):
            if i % 10 == 0:
                print i
            random_forest = RandomForestClassifier(**best_hyperparameters)
            random_forest = random_forest.fit(X_train, Y_train_class)
            local_fi = np.array(pd.DataFrame(random_forest.feature_importances_,
                                             index=X_train.columns,
                                             columns=['importance']).sort_values('importance', ascending=False).index)
            j = 1
            for feature in local_fi:
                feature_importance[feature] += j
                j += 1
        D = OrderedDict((k, v) for k, v in sorted(feature_importance.iteritems(), key=lambda kv: kv[1]))
        all_rankings[class_name] = D
        with open(DATA_PATH + '/feature_ranking_' + class_name + '.json', 'w') as f:
            f.write(json.dumps(D))
    return all_rankings


def extract_dataframe(feature_ranking, all_feature_rankings, number_of_items):
    columns = ['all', 'ECTODERM', 'NEURAL_CREST', 'MESODERM', 'ENDODERM']
    data_frame = pd.DataFrame(columns=columns)
    data_frame['all'] = feature_ranking.keys()[:number_of_items]
    data_frame['all'] = data_frame['all'].str.replace('proportion', 'prop')
    data_frame['all'] = data_frame['all'].str.replace('tumor_stage', 'ts')
    for label in LABELS:
        feature_ranking = all_feature_rankings[label]
        ordered_feature_ranking = OrderedDict(
            (k, v) for k, v in sorted(feature_ranking.iteritems(), key=lambda kv: kv[1]))
        data_frame[label] = ordered_feature_ranking.keys()[:number_of_items]
        data_frame[label] = data_frame[label].str.replace('proportion', 'prop')
        data_frame[label] = data_frame[label].str.replace('tumor_stage', 'ts')

    data_frame = data_frame.replace('donor_age_at_diagnosis', 'donor_age')
    return data_frame


def load_feature_ranking(label=''):
    if label == '':
        datapath = DATA_PATH + '/feature_ranking.json'
    else:
        datapath = DATA_PATH + '/feature_ranking_' + label + '.json'

    with open(datapath, 'r') as read_file:
        feature_ranking = json.loads(read_file.read())
    D = OrderedDict((k, v) for k, v in sorted(feature_ranking.iteritems(), key=lambda kv: kv[1]))
    return D


def best_n_features(n):
    try:
        feature_ranking = load_feature_ranking()
    except:
        print 'Generating the ranking...'
        X_train, Y_train, X_test, Y_test = load_data('dataset')
        best_hyperparameters = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 19, 'max_features': None,
                                'min_samples_leaf': 3, 'min_samples_split': 6, 'class_weight': 'balanced',
                                'n_estimators': 100}
        feature_ranking = feature_extractor(X_train, Y_train, best_hyperparameters, number_of_iterations=500)
    return feature_ranking.keys()[:n]


def best_n_features_one_vs_all(n, label):
    try:
        feature_ranking = load_feature_ranking(label)
    except:
        print 'Generating the ranking...'
        X_train, Y_train, X_test, Y_test = load_data('dataset')
        best_hyperparameters = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 19, 'max_features': None,
                                'min_samples_leaf': 3, 'min_samples_split': 6, 'class_weight': 'balanced',
                                'n_estimators': 100}
        all_rankings = feature_extractor_one_vs_all(X_train, Y_train, best_hyperparameters, number_of_iterations=500)
        feature_ranking = all_rankings[labels]

    feature_ranking = OrderedDict((k, v) for k, v in sorted(feature_ranking.iteritems(), key=lambda kv: kv[1]))
    return feature_ranking.keys()[:n]
