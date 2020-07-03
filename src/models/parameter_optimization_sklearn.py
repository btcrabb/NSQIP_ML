# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

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
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

global  ITERATION

def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']

    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold=N_FOLDS,
                        early_stopping_rounds=100, metrics='auc', seed=50)

    run_time = timer() - start

    # Extract the best score
    best_score = cv_results['auc-mean'][-1]

    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = len(cv_results['auc-mean'])

    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

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
    model = lgb.LGBMClassifier(**hyperparameters)

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

    features = pd.read_csv('../data/processed/NSQIP_Clean2.csv')

    # Extract the labels
    labels = np.array(features['READMISSION1'].astype(np.int32)).reshape((-1, ))
    features = features.drop(columns = ['READMISSION1', 'Unnamed: 0', 'index.1', 'index'])
    features.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in features.columns]

    # Split into training and testing data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 300, random_state = 0)

    print('Train shape: ', train_features.shape)
    print('Test shape: ', test_features.shape)

    model = lgb.LGBMClassifier(random_state=0)

    # Training set
    train_set = lgb.Dataset(train_features, label = train_labels)
    test_set = lgb.Dataset(test_features, label = test_labels)

    # Create a new file and open a connection
    OUT_FILE = '../../reports/optimization/bayes_test.csv'
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)

    # Write column names
    headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
    writer.writerow(headers)
    of_connection.close()

    # Create the algorithm
    tpe_algorithm = tpe.suggest

    # Record results
    trials = Trials()

    ITERATION = 0

    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
                max_evals = MAX_EVALS)

    bayes_results = evaluate(results, name = 'Bayesian')

    # Save the trial results
    with open('../../reports/optimization/trials.json', 'w') as f:
        f.write(json.dumps(trials_dict))

    print('Optimization Complete')

    return None


def main(argv):
    if len(sys.argv) == 2:

        # Define the search space
        space = {
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                        {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                        {'boosting_type': 'goss', 'subsample': 1.0}]),
            'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
            'is_unbalance': hp.choice('is_unbalance', [True, False]),
        }

        # Governing choices for search
        MAX_EVALS = sys.argv[2
        N_FOLDS = 5

        optimize()
    else:
        print('Usage: python parameter_optimization.py MAX_EVALS')


if __name__ == "__main__":
    main(sys.argv[1:])


