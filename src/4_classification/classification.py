#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

sys.path.insert(1, '../../src')

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import time
import pandas as pd
import itertools
import pydotplus
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
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
from sklearn import metrics, tree
from scipy import stats
from fancyimpute import MICE


# In[2]:


DATAPATH = '../../data_report'

try: os.mkdir(DATAPATH + '/plots')
except: pass
try: os.mkdir(DATAPATH + '/plots/confusion_matrix')
except: pass
try: os.mkdir(DATAPATH + '/datasets/clean')
except: pass
try: os.mkdir(DATAPATH + '/plots/trees')
except: pass
try: os.mkdir(DATAPATH + '/plots/feature_importance')
except: pass


# In[3]:


labels=['ECTODERM', 'NEURAL_CREST', 'MESODERM', 'ENDODERM']


# # Plot funtions

# In[4]:


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
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    plt.figure(figsize=(10, 12))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize=30,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plot_path = DATAPATH + '/plots/confusion_matrix/'
    plt.savefig(plot_path + title + name + '.png',bbox_inches='tight')

def plot_feature_importance(feature_importance, name):
    plt.clf()
    plt.figure(figsize=(20, 10))
    values0 = feature_importance.loc[feature_importance['importance'] > 0].index
    values1 = feature_importance.loc[feature_importance['importance'] > 0]['importance'].values

    plt.bar(values0[:10], values1[:10])
    plt.xticks(values0[:10], rotation=60)
    plt.title('Feature importance')
    plt.show()
    plt.savefig(DATAPATH + '/plots/feature_importance/' + 'feature_importance_' + name + '.png',bbox_inches='tight')

    
def plot_tree_graph(random_forest,columns,name):
    i = 0
    for tree_in_forest in random_forest.estimators_[:1]:
        # Create DOT data
        dot_data = tree.export_graphviz(tree_in_forest, out_file=None,
                                        feature_names=list(columns),
                                        class_names=random_forest.classes_)
        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)

        # Show graph
        graph.create_png()
        graph.write_png(DATAPATH + '/plots/trees' + '/' + name + ".png")
        i += 1


# # Preprocessing

# In[7]:


def nan_imputing(df):
    """
    There is only one feature with nans. Donor age at diagnosis. 
    We impute it using the MICE strategy
    :param df:
    :return:
    """
    # Imput missing data with mice
    fancy_imputed = df
    dummies = pd.get_dummies(df)
    imputed = pd.DataFrame(data=MICE(verbose=False).complete(dummies), columns=dummies.columns, index=dummies.index)
    fancy_imputed.donor_age_at_diagnosis = imputed.donor_age_at_diagnosis
    fancy_imputed['donor_age_at_diagnosis'] = fancy_imputed['donor_age_at_diagnosis'].astype(np.int)
    return fancy_imputed

def preprocessing(df):
    y = df.pop('histology_tier1')
    X = df.drop(['Unnamed: 0', 'histology_tier2'], axis=1)
    X['donor_sex'] = X['donor_sex'].str.replace('female','1')
    X['donor_sex'] = X['donor_sex'].str.replace('male','0')

    X['female'] = pd.to_numeric(X['donor_sex'])
    
    X = X.drop('donor_sex',axis=1)
    X_train, X_test, Y_train, Y_test =         train_test_split(pd.get_dummies(X), y, stratify=y, test_size=.2, random_state=42)
    X_train = nan_imputing(X_train)
    X_test = nan_imputing(X_test)
    X_train['number_of_breaks'] = X_train['DUP'] + X_train['DEL'] + X_train['TRA'] + X_train['h2hINV'] +                                   X_train['t2tINV']
    X_test['number_of_breaks'] = X_test['DUP'] + X_test['DEL'] + X_test['TRA'] + X_test['h2hINV'] + X_test[
        't2tINV']
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

    X_train.head()
    return X_train, Y_train, X_test, Y_test


# # SVM

# In[ ]:


def try_svm(X_train, Y_train, X_test, Y_test, name='with_max_cc',
                               path=DATAPATH + '/plots/trees'):
    from sklearn.svm import SVC
    import random
    n_iter_search = 5
    # f = open('../../data/best_params' + 'random_forest' + name + '.txt', 'w')
    param_dist = {"C": stats.uniform(0, 1),
                  "kernel": ['rbf', 'linear', 'poly', 'sigmoid'],
                  "decision_function_shape": ['ovo', 'ovr']
                  }
    labels = ['ECTODERM', 'NEURAL_CREST', 'MESODERM', 'ENDODERM']
    clf = SVC(class_weight='balanced')

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, iid=False,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)
    random_search.fit(X_train, Y_train.values.ravel())
    print('trained')
    score = random_search.score(X_test, Y_test)

    # f.write('Random Forest ' + str(score))
    print 'SVM', score
    y_test_pred = random_search.predict(X_test)
    # Compute confusion matrix
    class_names = labels
    cnf_matrix = metrics.confusion_matrix(Y_test, y_test_pred,
                                          labels=class_names)

    print(cnf_matrix)


# # Random forest feature selection and classification

# In[11]:


def feature_extraction_and_classification_rf(X_train, y_train, X_test, y_test, name, n_iter_search = 50,class_names=labels):
    # Random forest
   
    param_dist = {"max_depth": stats.randint(2, 20),
                  "min_samples_split": stats.randint(2, 11),
                  "min_samples_leaf": stats.randint(1, 20),
                  "bootstrap": [True, False],
                  "max_features": ['auto', 'log2', None],
                  "criterion": ["gini", "entropy"]}

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, iid=False,
                                       n_iter=n_iter_search, pre_dispatch=3, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())

    best_params = random_search.best_params_
    best_params['n_estimators'] = 100
    best_params['class_weight'] = 'balanced'
    print 'Best params', best_params
    random_forest = RandomForestClassifier(**best_params)

    random_forest = random_forest.fit(X_train, y_train)

    # plot the graph
    plot_tree_graph(random_forest,X_train.columns,name)
    
    # plot the feature importance
    feature_importances = pd.DataFrame(random_forest.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    plot_feature_importance(feature_importances, name)
    
    # plot the classification results
    score = random_forest.score(X_test, y_test)

    print 'Random Forest', score
    y_test_pred = random_search.predict(X_test)
    # Compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred,
                                          labels=class_names)

    print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization', name=name)

    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix', name=name)
    
    return feature_importances


# In[ ]:


def run_feature_extractor(dataset_files_paht,n_iter):
    """
    :param dataset_files_paht: array[string] array of the paths of the datasets to test
    :param n_iter: number of iterations on the crossvalidation
    :return: 
    """
    for path in dataset_files_paht:
        try:
            try: 
                X_train = pd.read_csv(dataset_file.replace('.csv', '_clean.csv'),index_col=0)
                Y_train = X_train.pop('histology_tier1')
                X_test = pd.read_csv(dataset_file.replace('.csv', '_test_clean.csv'),index_col=0)
                Y_test = X_test.pop('histology_tier1')
                print 'Dataset', dataset_file.replace('.csv', '_clean.csv')    
            except:
                print 'Dataset', dataset_file        
                df = pd.read_csv(dataset_file)
                print 'Preprocessing dataset:', dataset_file
                X_train, Y_train, X_test, Y_test = preprocessing(df)
            
            print 'Running feature extractor..'
            feature_importance = feature_extraction_and_classification_rf(X_train, Y_train, X_test, Y_test, name='find_best_hyper', n_iter_search = n_iter)
            # save the clean dataset for revise it is ok
            X_train['histology_tier1'] = Y_train
            X_test['histology_tier1'] = Y_test
            X_train.to_csv(path.replace('.csv', '_clean.csv'))
            X_test.to_csv(path.replace('.csv', '_test_clean.csv'))
        except Exception as e:
            print('This path is not working:', path)
            print('Error:',e)
    return feature_importance


# # One vs All

# In[ ]:


def one_vs_all_random_forest(dataset_file,meta=True):
    print 'Dataset', dataset_file

    for class_name in labels:

        print 'One vs all ', class_name
        to_remove = [c for c in labels if c != class_name]

        df = pd.read_csv(dataset_file)
        df = df.replace(to_replace=to_remove, value='OTHER')
        print 'Preprocessing dataset:', dataset_file

        X_train, Y_train, X_test, Y_test = preprocessing(df)
        print(X_train.columns)
        feature_extraction_and_classification_rf(X_train, Y_train, X_test, Y_test, name='report', n_iter_search = 50,class_names=[class_name, 'OTHER'])

    if not meta:    
        for class_name in labels:

            print 'One vs all no meta', class_name
            to_remove = [c for c in labels if c != class_name]

            df = pd.read_csv(path)
            df = df.drop(['donor_age_at_diagnosis', 'donor_sex', 'tumor_stage1', 'tumor_stage2'], axis=1)
            df = df.replace(to_replace=to_remove, value='OTHER')
            X_train, Y_train, X_test, Y_test = preprocessing(df)
            feature_extraction_and_classification_rf(X_train, Y_train, X_test, Y_test, name='report', n_iter_search = 50)


# # Best params calculation
# We obtained: 
# Best params {'bootstrap': True, 'min_samples_leaf': 3, 'n_estimators': 100, 'min_samples_split': 5, 'criterion': 'entropy', 'max_features': None, 'max_depth': 13, 'class_weight': 'balanced'}

# In[12]:


datasets = ['../../data_report/datasets/dataset_final.csv']
# first run to find the best hyperparameters
feature_importance = run_feature_extractor(datasets,500)
feature_importance


# # Feature selection: 
# We fix the hyperparameters and train a random forest k times. Then we sum the positions of the features and return them as a dictionary

# In[ ]:


def voting_feature_selection(dataset_file,k,best_params,features):
    # k= number of votations
    # Find the order of every feature
    try: 
        X_train = pd.read_csv(dataset_file.replace('.csv', '_clean.csv'),index_col=0)
        Y_train = X_train.pop('histology_tier1')
        print 'Dataset', dataset_file.replace('.csv', '_clean.csv')    
    except:
        print 'Dataset', dataset_file        
        df = pd.read_csv(dataset_file)
        print 'Preprocessing dataset:', dataset_file
        X_train, Y_train, X_test, Y_test = preprocessing(df)
    feature_importance = {feature:0 for feature in features}
    for i in range(k):
        if i%10==0:
            print i
        random_forest = RandomForestClassifier(**best_params)
        random_forest = random_forest.fit(X_train, Y_train)
        local_fi = np.array(pd.DataFrame(random_forest.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False).index)
        j = 1
        for feature in local_fi:
            feature_importance[feature] += j
            j +=1
    return feature_importance


# In[ ]:


best_params = {'bootstrap': True, 'min_samples_leaf': 3, 'n_estimators': 100, 'min_samples_split': 5, 'criterion': 'entropy', 'max_features': None, 'max_depth': 13, 'class_weight': 'balanced'}
features = np.array(feature_importance.index)
k = 300
fi_dict = voting_feature_selection(datasets[0],k,best_params,features)
fi_dict


# In[ ]:


sorted_by_value = sorted(fi_dict.items(), key=lambda kv: kv[1])


# In[30]:


def main():
    # datasets = os.listdir(DATAPATH + '/datasets/')
    datasets = ['../../data_report/datasets/dataset_final.csv']
    print('Runing feature extractor')
    #run_feature_extractor(datasets)
    one_vs_all_random_forest(datasets[0])


if __name__ == '__main__':
    init = time.time()
    main()
    print'time:', timedelta(seconds=time.time() - init)

