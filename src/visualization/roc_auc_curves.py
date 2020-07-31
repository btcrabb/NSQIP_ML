import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

# Define a random seed for reproducibility
seed = 0
np.random.seed(seed)

# utils
from util import util

import sklearn

import os
import pickle
import matplotlib.pyplot as plt

from options.analysis_options import AnalysisOptions

def bootstrap_roc_curves(model, X_test, Y_test):
    """
    Bootstrap the auc values for the ROC curves
    :param model: an sklearn model
    :param X_test: the testing features
    :param Y_test: the testing labels
    :return: false positives, true positives, and bootstrapped auc scores
    """

    # predict probabilities
    lr_probs = model.predict_proba(X_test)

    # keep probabilities for the positive outcome only
    lr_probs = [round(x, 2) for x in lr_probs[:, 1]]

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    fprList = []
    tprList = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(lr_probs), len(lr_probs))
        if len(np.unique(np.array(Y_test)[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        labels = np.array(Y_test)[indices]
        predictions = np.array(lr_probs)[indices]

        # calculate scores and curve
        bootstrapped_scores.append(sklearn.metrics.roc_auc_score(labels, predictions))
        lr_fpr, lr_tpr, _ = sklearn.metrics.roc_curve(labels, predictions, drop_intermediate=True)

        fprList.append(lr_fpr)
        tprList.append(lr_tpr)

    return fprList, tprList, bootstrapped_scores


def generate_roc_curves(models, names, X_test, Y_test, outfile='reports/figures/ROC_AUC_comparison.png', bootstrap=True, save=False):
    """
    Generate ROC curves and calculate AUC scores
    :param models: list of models to generate curves for
    :param names: names of the models
    :param X_test: the testing features
    :param Y_test: the testing labels
    :param bootstrap: perform bootstrapping of AUC scores
    :param save: save the ROC curve figure
    :return: list of AUC scores, list of bootstrapped standard deviations
    """

    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('tab10')

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(Y_test))]

    aucList = []
    stdList = []

    for i, model in enumerate(models):

        # predict probabilities
        lr_probs = model.predict_proba(X_test)

        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]

        # calculate scores and curve
        lr_auc = sklearn.metrics.roc_auc_score(Y_test, lr_probs)
        lr_fpr, lr_tpr, _ = sklearn.metrics.roc_curve(Y_test, lr_probs)

        if bootstrap:
            fpr, tpr, scores = bootstrap_roc_curves(model, X_test, Y_test)

            scores.sort()
            sorted = np.array(scores)

            aucList.append([lr_auc, sorted[25], sorted[975]])
            stdList.append(np.std(sorted))

            # plot the roc curve for the model
            plt.plot(lr_fpr, lr_tpr, color=cmap(0.1 * i), marker='',
                     label='{} (AUC = {} (95% CI {} to {}))'.format(names[i], round(lr_auc, 3), round(sorted[25], 3),
                                                                    round(sorted[975], 3)))
        else:
            # plot the roc curve for the model
            plt.plot(lr_fpr, lr_tpr, color=cmap(0.1 * i), marker='', label='{} (AUC = {})'.format(names[i], round(lr_auc, 3)))

    # plot the roc curve for the no skill model
    ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(Y_test, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', color='black', label='No Skill (AUC = 0.500)')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # title
    plt.title('ROC AUC Scores by Algorithm Type')
    # show the legend
    plt.legend(loc='lower right')

    # save the figure
    if save:
        plt.savefig(outfile)

    # show the plot
    plt.show()

    return aucList, stdList

def main():

    # parse options
    opt = AnalysisOptions().parse()

    # load training and testing datasets
    X_train, Y_train, X_test, Y_test = util.load_train_and_test(opt.train_test_datasets_path)

    # load all optimized models from the models folder
    if os.path.isdir(opt.models):
        dirName = opt.models

        fileList = list()

        for (dirpath, dirnames, filenames) in os. walk(dirName):
            for file in filenames:
                if '.sav' in file:
                    fileList.append(os.path. join(dirpath, file))

        modelList = list()

        for file in fileList:
            model = pickle.load(open(file, 'rb'))
            modelList.append(model)

    else:
        modelList = list()
        fileList = [opt.models]
        model = pickle.load(open(opt.models, 'rb'))
        modelList.append(model)


    index = [file.split('/')[-1].rstrip('.sav') for file in fileList]
    generate_roc_curves(modelList, index, X_test, Y_test, outfile=opt.outfile, bootstrap=opt.bootstrap, save=opt.save)


if __name__ == "__main__":
    main()