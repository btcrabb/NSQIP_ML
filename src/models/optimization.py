import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Data manipulation
import pandas as pd
import numpy as np
from datetime import date
import argparse
import pickle
import sys
import os
import sklearn
from collections import MutableMapping
from util import util
from options.optimization_options import OptimizationOptions

# Modeling
from sklearn import model_selection

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import ast

# Optimization
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""

    global ITERATION
    # Keep track of evals
    ITERATION += 1

    # Extract the lower level keys
    hyperparameters = flatten(hyperparameters)
    
    # convert necessary parameters to tuples
    if 'hidden_layer_size' in hyperparameters:
        hyperparameters['hidden_layer_sizes'] = tuple([int(hyperparameters['hidden_layer_sizes'])])

    start = timer()
	
    # Make sure parameters that need to be integers are integers
    if type(MODEL).__name__ == 'XGBClassifier':
        hyperparameters['max_depth'] = int(hyperparameters['max_depth'])
        hyperparameters['n_estimators'] = int(hyperparameters['n_estimators'])

    # define model
    model = MODEL.set_params(**hyperparameters)
    # define scoring method
    if METRIC == 'f1_score':
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score)
    elif METRIC == 'balanced_accuracy_score':
         scorer = sklearn.metrics.make_scorer(sklearn.metrics.balanced_accuracy_score) 
    else:
         scorer = sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)

    # Perform n_folds cross validation
    kfold = model_selection.KFold(n_splits=N_FOLDS)
    cv_results = model_selection.cross_validate(model, train_features, train_labels, scoring=scorer, cv=kfold)

    run_time = timer() - start

    # Extract the best score
    best_score = np.mean(cv_results['test_score'])

    # Loss must be minimized
    loss = 1 - best_score

    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


def evaluate(results, name):
    """Evaluate model on test data using hyperparameters in results
       Return dataframe of hyperparameters"""

    new_results = results.copy()
    # String to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)

    # Sort with best values on top
    new_results = new_results.sort_values('score', ascending=False).reset_index(drop=True)

    # Print out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name,
                                                                                                new_results.loc[
                                                                                                    0, 'score'],
                                                                                                new_results.loc[
                                                                                                    0, 'iteration']))

    # Use best hyperparameters to create a model
    hyperparameters = new_results.loc[0, 'hyperparameters']
    print('Optimal Hyperparameters: {}'.format(hyperparameters))

    if type(MODEL).__name__ == 'SVC':
        model = MODEL.set_params(**hyperparameters, probability=True)
    else:
        model = MODEL.set_params(**hyperparameters)

    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]

    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))

    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns=list(new_results.loc[0, 'hyperparameters'].keys()))
    
    for i, row in enumerate(new_results['hyperparameters']):
        if 'hidden_layer_sizes' in row:
            new_results['hyperparameters'][i]['hidden_layer_sizes'] = str(new_results['hyperparameters'][i]['hidden_layer_sizes'])

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index=[0]),
                               ignore_index=True)

    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']

    return hyp_df

def optimize(space):

    # Split into training and testing data
    global train_features
    global test_features
    global train_labels
    global test_labels

    train_features, train_labels, test_features, test_labels = util.load_train_and_test(DATASET_PATH)

    train_features = train_features
    test_features = test_features

    # squeeze label dimensions
    test_labels = test_labels.values.ravel()
    train_labels = train_labels.values.ravel()

    if os.path.exists(OUT_FILE) and CONTINUE==1:
        print('Using {}'.format(OUT_FILE))
    else:
	     # Create a new file and open a connection
        of_connection = open(OUT_FILE, 'w')
        writer = csv.writer(of_connection)
        
        # Write column names
        headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
        writer.writerow(headers)
        of_connection.close()

    global trials 
    global ITERATION
	
    # Record results
    if CONTINUE == 1:
        pfile = OUT_FILE.rstrip('.csv') + '_trials.p'
        with open(pfile, 'rb') as f:
            trials = pickle.load(f)
            ITERATION = len(trials)
            print('Starting on iteration {}'.format(ITERATION))
    else:
        trials = Trials()
        ITERATION = 0

    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
                max_evals = MAX_EVALS)

    results = pd.read_csv(OUT_FILE)
    bayes_results = evaluate(results, name = 'Bayesian').infer_objects()
    best_bayes_params = bayes_results.iloc[bayes_results['score'].idxmax(), :].copy()

    # DataFrame of just scores
    scores = pd.DataFrame({'{}'.format(METRIC): bayes_results['score'], 'iteration': bayes_results['iteration'], 'search': 'Bayesian'})

    scores['{}'.format(METRIC)] = scores['{}'.format(METRIC)].astype(np.float)
    scores['iteration'] = scores['iteration'].astype(np.int)

    # Plot of scores over the course of searching
    sns.lmplot('iteration', '{}'.format(METRIC), hue = 'search', data = scores, aspect = 2, scatter_kws={"s": 5})
    plt.scatter(best_bayes_params['iteration'], best_bayes_params['score'], marker = '*', s = 200, c = 'orange', edgecolor = 'k')
    plt.xlabel('Iteration'); plt.ylabel('{}'.format(METRIC))
    plt.savefig(OUT_FILE.rstrip('.csv') + '_SCORES.png')

    # Save the trial results
    pfile = OUT_FILE.rstrip('.csv') + '_trials.p'
    with open(pfile, 'wb') as f:
        pickle.dump(trials, f)

    print('Optimization Complete')

    return None

def flatten(d, parent_key ='', sep ='_'): 
    items = [] 
    for k, v in d.items(): 
        new_key = k if parent_key else k 
  
        if isinstance(v, MutableMapping): 
            items.extend(flatten(v, new_key, sep = sep).items()) 
        else: 
            items.append((new_key, v)) 
    return dict(items)


def main(**kwargs):

    if not kwargs:
        opt = OptimizationOptions().parse()
    else:
        opt = OptimizationOptions().parse()
        opt.model = kwargs.get('model')
        opt.outfile = '../../reports/optimization/{}/{}_bayes_test.csv'.format(opt.model, date.today())

        print('Running optimization for {}'.format(opt.model))

    # Governing choices for search
    global MAX_EVALS
    global N_FOLDS
    global CONTINUE
    global OUT_FILE
    global METRIC
    global DATASET_PATH

    MAX_EVALS = opt.max_evals
    N_FOLDS = opt.n_folds
    CONTINUE = opt.continue_opt
    OUT_FILE = opt.outfile
    METRIC = opt.metric
    DATASET_PATH = opt.dataset_path
    
    global MODEL
    if opt.model == 'SVC':

        # load SVC model
        from sklearn.svm import SVC
        MODEL = SVC()
        
        # Define the search space
        space = {
            'kernel': hp.choice('kernel',
                               [{'kernel': 'linear'},
                                {'kernel': 'rbf', 'gamma': hp.choice('rbf_gamma', ['scale', 'auto'])},
                                {'kernel': 'poly', 'gamma': hp.choice('poly_gamma', ['scale', 'auto'])},
                                {'kernel': 'sigmoid', 'gamma': hp.choice('sigmoid_gamma', ['scale', 'auto'])}]),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'C': hp.uniform('C', 0, 10.0),
            'tol': hp.uniform('tol', 0.0000000001, 0.01)
        }

    elif opt.model == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        MODEL = DecisionTreeClassifier()

        # Define the search space
        space = {
            'criterion': hp.choice('criterion',
                                 [{'criterion': 'gini'},
                                  {'criterion': 'entropy'}]),
            'min_samples_split': hp.choice('min_samples_split', range(2,10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'random_state': hp.choice('random_state', [0]),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5),
            'class_weight': hp.choice('class_weight', ['balanced', None])
        }

    elif opt.model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        MODEL = RandomForestClassifier()

        # Define the search space
        space = {
            'n_estimators': hp.choice('n_estimators', range(10,300)),
            'criterion': hp.choice('criterion',
                                 [{'criterion': 'gini'},
                                  {'criterion': 'entropy'}]),
            'min_samples_split': hp.choice('min_samples_split', range(2,10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'random_state': hp.choice('random_state', [0]),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5),
            'n_jobs': hp.choice('n_jobs', [8]),
            'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
        }

    elif opt.model == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier
        MODEL = AdaBoostClassifier()

        # Define the search space
        space = {
            'n_estimators': hp.choice('n_estimators', range(10,100)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 10),
            'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
            'random_state': hp.choice('random_state', [0]),
        }

    elif opt.model == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingClassifier
        MODEL = GradientBoostingClassifier()

        # Define the search space
        space = {
            'loss': hp.choice('loss', ['deviance', 'exponential']),
            'learning_rate': hp.uniform('learning_rate', 0.01, 10),
            'n_estimators': hp.choice('n_estimators', range(10,300)),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'criterion': hp.choice('criterion',
                                 [{'criterion': 'friedman_mse'},
                                  {'criterion': 'mse'},
                                  {'criterion': 'mae'}]),
            'min_samples_split': hp.choice('min_samples_split', range(2,10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'max_depth': hp.choice('max_depth', range(1,30)),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'random_state': hp.choice('random_state', [0]),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5),
        }

    elif opt.model == 'XGBoost':
        import xgboost

        MODEL = xgboost.XGBClassifier()

        # Define the search space
        space = {
            'booster': hp.choice('booster',
                                 [{'booster': 'gbtree', 'subsample': hp.uniform('tree_subsample', 0.5, 1)},
                                  {'booster': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)}]),
            'learning_rate': hp.loguniform('eta', np.log(0.01), np.log(0.5)),
            'gamma': hp.uniform('gamma', 0.0, 10),
            'max_depth': hp.quniform('max_depth', 1, 20, 1),
            'min_child_weight': hp.uniform('min_child_weight', 0, 10),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100),
            'n_estimators': hp.quniform('n_estimators', 1, 200, 1),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
        }
        
    elif opt.model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        
        MODEL = MLPClassifier()
        
        def hidden_layers(depth, max_nodes1, max_nodes2):
            options = []
            for k in range(1, max_nodes1):
                for i in range(0,depth):
                    if i == 0:
                        options.append(tuple([k,]))
                    else:
                        for j in range(1, max_nodes2):
                            options.append(tuple([k, j]))
            return options
        
        layers = [(4,), (8,), (16,), (24,), (32,), (48,), (64,), (128,), (4,2), (8,4), (16,8), (24,12), (32,16), (48,24), (64,32), (128,64), (128,64,32)]
        
        # Define the search space
        space = {
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', layers),
            # 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', range(2, 100, 1)),
            'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': hp.choice('solver', [{'solver': 'lbfgs'},
                                           {'solver': 'sgd', 
                                                'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive'])},
                                           {'solver': 'adam'}]),
            'alpha': hp.uniform('alpha', 0, 0.001),
            'batch_size': hp.choice('batch_size', range(1, 100, 1)),
            'learning_rate_init': hp.uniform('learning_rate_init', 0.00000001, 0.1),
            'power_t': hp.uniform('power_t', 0.1, 1),
            'max_iter': hp.choice('max_iter', range(100,300)),
            'tol': hp.uniform('tol', 0.0000001, 0.01),
            'early_stopping': hp.choice('early_stopping', [True]),
            'momentum': hp.uniform('momentum', 0.5, 0.99),
            'beta_1': hp.uniform('beta_1', 0.5, 0.9999),
            'beta_2': hp.uniform('beta_2', 0.9, 0.99999)
        }
        
    else:
        print('Unknown model')
        exit
    

    optimize(space)


if __name__ == "__main__":
    main()