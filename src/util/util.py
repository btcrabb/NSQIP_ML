"""This module contains simple helper functions """
from __future__ import print_function
import pandas as pd
import pickle
import os
import ast

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_train_and_test(path):

    """Loads training features, training labels, testing features, and testing features
    Parameters:
        path (str) -- a single directory path containing all four datasets
    """

    train_features = pd.read_csv(path + 'train_features.csv', index_col=0)
    train_labels = pd.read_csv(path + 'train_labels.csv', index_col=0)
    test_features = pd.read_csv(path + 'test_features.csv', index_col=0)
    test_labels = pd.read_csv(path + 'test_labels.csv', index_col=0)

    return train_features, train_labels, test_features, test_labels


def load_model(name, hyperparameters):
    """ Loads the appropriate sklearn model from a model name and hyperparameters
    Parameters:
        name (str) -- the name of the model to load
        hyperparameters -- the hyperparameters to load the model with
        """

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    if name == 'AdaBoost':
        model = AdaBoostClassifier(**hyperparameters)
    elif name == 'DecisionTree':
        model = DecisionTreeClassifier(**hyperparameters)
    elif name == 'KMeans':
        model = KNeighborsClassifier(**hyperparameters)
    elif name == 'MLP':
        model = MLPClassifier(**hyperparameters)
    elif name == 'RandomForest':
        model = RandomForestClassifier(**hyperparameters)
    elif name == 'SVC':
        model = SVC(**hyperparameters, probability=True)
    elif name == 'XGBoost':
        model = XGBClassifier(**hyperparameters)
    else:
        print('Unkown model name')

    return model


def save_models(dirName):
    """ Saves trained models using hyperparameters from hyperopt optimization file
    Parameters:
        dirName (str) the directory containing optimization files
    """
    # load a list of all optimization results files
    fileList = list()

    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if '.csv' in file:
                fileList.append(os.path.join(dirpath, file))

    # load datasets
    train_features, train_labels, test_features, test_labels = load_train_and_test('data/split/')

    for file in fileList:
        # load hyperparameter optimization files
        results = pd.read_csv(file)

        # find the model_name from the filename
        file_split = file.split('/')[3]
        model_name = file_split.split('\\')[0]

        new_results = results.copy()

        # String to dictionary
        new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)

        # Sort with best values on top
        new_results = new_results.sort_values('score', ascending=False).reset_index(drop=True)

        # Use best hyperparameters to create a model
        hyperparameters = new_results.loc[0, 'hyperparameters']

        # load the appropriate model and fit on training data
        model = load_model(model_name, hyperparameters)
        model.fit(train_features, train_labels)

        # create output filename
        new_dir = 'models/'
        filename = new_dir + model_name + '.sav'

        # save the trained model
        pickle.dump(model, open(filename, 'wb'))