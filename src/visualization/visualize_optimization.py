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


def evaluate(results, name):
    """Return dataframe of hyperparameters"""
    
    new_results = results.copy()
    
    # String to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)
    
    # Sort with best values on top
    new_results = new_results.sort_values('score', ascending = False).reset_index(drop = True)
    
    # Print out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, 
                                        new_results.loc[0, 'score'], new_results.loc[0, 'iteration']))
    
    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns = list(new_results.loc[0, 'hyperparameters'].keys()))
    
    for i, row in enumerate(new_results['hyperparameters']):
        if 'hidden_layer_sizes' in row:
            new_results['hyperparameters'][i]['hidden_layer_sizes'] = str(new_results['hyperparameters'][i]['hidden_layer_sizes'])

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
                               ignore_index = True)
        
    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']
    
    return hyp_df


def main(**kwargs):

    if not kwargs:
        parser = argparse.ArgumentParser(description='Optimize Hyperparameters')
        parser.add_argument('--outdir', '-o', type=str, required=False,
                            default='../../reports/optimization/',
                            help='the directory to save all generated figures and files')
        parser.add_argument('--results', '-r', type=str, required=False,
                            default='../../reports/optimization/bayes_test.csv',
                            help='the results.csv file')

        args = parser.parse_args()


        # Parse arguments
        OUT_FILE = args.outdir
        RESULTS = args.results
    else:
        OUT_FILE = kwargs.get('out_file')
        RESULTS = kwargs.get('results')
    
    bayes_results = pd.read_csv(RESULTS).sort_values('score', ascending = False).reset_index()
    bayes_params = evaluate(bayes_results, name = 'Bayesian').infer_objects()
    
    # Dataframe of just scores
    scores = pd.DataFrame({'ROC AUC': bayes_params['score'], 'iteration': bayes_params['iteration'], 'search': 'Bayesian'})

    scores['ROC AUC'] = scores['ROC AUC'].astype(np.float32)
    scores['iteration'] = scores['iteration'].astype(np.int32)
    
    best_bayes_params = bayes_params.iloc[bayes_params['score'].idxmax(), :].copy()
    
    hypers = bayes_params.select_dtypes(include=[np.number]).columns.drop('score', 'iteration').values
    rows = int(np.ceil(len(hypers)/4))

    # Plot of scores over the course of searching
    sns.lmplot('iteration', 'ROC AUC', hue = 'search', data = scores, aspect = 2, scatter_kws={"s": 5});
    plt.scatter(best_bayes_params['iteration'], best_bayes_params['score'], marker = '*', s = 200, c = 'orange', edgecolor = 'k')
    plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("Validation ROC AUC versus Iteration");
    plt.savefig(OUT_FILE + 'scores_by_iteration.png')
    
    for i, hyper in enumerate(hypers):
        plt.figure(figsize=(12,6))
        sns.regplot('iteration', str(hyper), data = bayes_params)
        plt.scatter(best_bayes_params['iteration'], best_bayes_params[str(hyper)], marker = '*', s = 200, c = 'k')
        plt.xlabel('Iteration')
        plt.ylabel('{}'.format(hyper))
        plt.title('{} over Search'.format(hyper))
        plt.tight_layout()
        plt.savefig(OUT_FILE + '{}_by_iteration.png'.format(hyper))      

    

if __name__ == "__main__":
    main()