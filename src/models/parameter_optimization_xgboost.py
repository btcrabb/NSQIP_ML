# Data manipulation
import pandas as pd
import numpy as np
from datetime import date
import argparse
import pickle
import sys
import os
import sklearn

# Modeling
import lightgbm as lgb
import xgboost
from sklearn import model_selection

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, make_scorer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import ast

# Optimization
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin


def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""

    global ITERATION
    # Keep track of evals
    ITERATION += 1

    # Retrieve the subsample
    subsample = hyperparameters['booster'].get('subsample', 1.0)

    # Extract the boosting type and subsample to top level keys
    hyperparameters['booster'] = hyperparameters['booster']['booster']
    hyperparameters['subsample'] = subsample

    start = timer()
	
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['max_depth', 'n_estimators']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    # define model
    model = xgboost.XGBClassifier(**hyperparameters, verbosity=0)
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
    model = xgboost.XGBClassifier(**hyperparameters)

    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]

    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))

    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns=list(new_results.loc[0, 'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index=[0]),
                               ignore_index=True)

    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']

    return hyp_df

def optimize(space):

    features = pd.read_csv('../../data/processed/NSQIP_Clean2.csv')
    #limfeats = ['READMISSION1','RETURNOR', 'PRBUN', 'PRSODM', 'PRINR', 'OPTIME', 'PRPTT', 'AGE', 'PRCREAT', 'BLEEDIS_1.0', 'PRPLATE', 'WTLOSS', 'PRPT', 'SMOKE']
    #features = features[limfeats]
    features['AGE'].replace(np.NaN, features['AGE'].median(), inplace=True)
    features = features.dropna()

    # Extract the labels
    labels = np.array(features['READMISSION1'].astype(np.int32)).reshape((-1, ))
    features = features.drop(columns = ['READMISSION1', 'Unnamed: 0', 'index.1', 'index'])

    features.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in features.columns]

    # Split into training and testing data
    global train_features
    global test_features
    global train_labels
    global test_labels

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 300, random_state = 0)

    print('Train shape: ', train_features.shape)
    print('Test shape: ', test_features.shape)

    model = xgboost.XGBClassifier()

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

    # Create the algorithm
    tpe_algorithm = tpe.suggest

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

    # Dataframe of just scores
    scores = pd.DataFrame({'{}'.format(METRIC): bayes_results['score'], 'iteration': bayes_results['iteration'], 'search': 'Bayesian'})

    scores['{}'.format(METRIC)] = scores['{}'.format(METRIC)].astype(np.float32)
    scores['iteration'] = scores['iteration'].astype(np.int32)
	
    # Plot of scores over the course of searching
    sns.lmplot('iteration', '{}'.format(METRIC), hue = 'search', data = scores, aspect = 2, scatter_kws={"s": 5});
    plt.scatter(best_bayes_params['iteration'], best_bayes_params['score'], marker = '*', s = 200, c = 'orange', edgecolor = 'k')
    plt.xlabel('Iteration'); plt.ylabel('{}'.format(METRIC))
    plt.savefig(OUT_FILE.rstrip('.csv') + '_SCORES.png')

    # Save the trial results
    pfile = OUT_FILE.rstrip('.csv') + '_trials.p'
    with open(pfile, 'wb') as f:
        pickle.dump(trials, f)

    print('Optimization Complete')

    return None


def main(argv):

    parser = argparse.ArgumentParser(description='Optimize Hyperparameters')
    parser.add_argument('--max_evals', '-m', type=int, required=False, default=10,
                        help='the maximun number of evaluations to run')
    parser.add_argument('--continue_opt', '-c', type=int, required=False, default=0,
                        help='continue optimization from previous trials')
    parser.add_argument('--n_folds', '-n', type=int, required=False, default=5,
                        help='the number of cross validation folds')
    parser.add_argument('--outfile', '-o', type=str, required=False,
                        default='../../reports/optimization/{}_bayes_test.csv'.format(date.today()),
                        help='the number of cross validation folds')
    parser.add_argument('--metric', '-s', type=str, required=False,
                        default='balanced_accuracy_score',
                        help='the metric used to assess cross validation performance')

    args = parser.parse_args()
    # Define the search space
    space = {
        'booster': hp.choice('booster',
                                   [{'booster': 'gbtree', 'subsample': hp.uniform('tree_subsample', 0.5, 1)},
                                    {'booster': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                    {'booster': 'gblinear', 'subsample': hp.uniform('linear_subsample', 0.5, 1)}]),
        'learning_rate': hp.loguniform('eta', np.log(0.01), np.log(0.5)),
        'gamma': hp.uniform('gamma', 0.0, 10),
        'max_depth': hp.quniform('max_depth', 1, 20, 1),
        'min_child_weight': hp.uniform('min_child_weight', 0, 10),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100),
        'min_child_weight': hp.uniform('min_child_samples', 0, 100),
        'n_estimators': hp.quniform('n_estimators', 1, 200, 1),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
    }

    # Governing choices for search
    global MAX_EVALS
    global N_FOLDS
    global CONTINUE
    global OUT_FILE
    global METRIC

    MAX_EVALS = args.max_evals
    N_FOLDS = args.n_folds
    CONTINUE = args.continue_opt
    OUT_FILE = args.outfile
    METRIC = args.metric

    optimize(space)


if __name__ == "__main__":
    main(sys.argv[1:])


