import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import tree
import pydotplus

DATA_PATH = '../definitive_data_folder'
PLOT_PATH = DATA_PATH + '/plots'

try: os.mkdir(PLOT_PATH)
except: pass
try: os.mkdir(PLOT_PATH + '/trees')
except: pass
try: os.mkdir(PLOT_PATH + '/feature_importance')
except: pass

def plot_confusion_matrix(cm, classes,
                          normalize=True,
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
    if len(classes) < 5:
        plt.figure(figsize=(10, 12))
        font = 30
    else:
        plt.figure(figsize=(20, 22))
        font = 20

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=font,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plot_path = PLOT_PATH + '/'
    plt.savefig(plot_path + title + name + '.png', bbox_inches='tight')


def plot_feature_importance(feature_importance, name):
    plt.clf()
    plt.figure(figsize=(20, 10))
    values0 = feature_importance.loc[feature_importance['importance'] > 0].index
    values1 = feature_importance.loc[feature_importance['importance'] > 0]['importance'].values

    plt.bar(values0[:10], values1[:10])
    plt.xticks(values0[:10], rotation=60)
    plt.title('Feature importance')
    plt.show()
    plt.savefig(PLOT_PATH + '/feature_importance/' + 'feature_importance_' + name + '.png', bbox_inches='tight')


def plot_tree_graph(random_forest, columns, name):
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
        graph.write_png(PLOT_PATH + '/trees' + '/' + name + ".png")
        i += 1