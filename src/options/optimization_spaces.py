import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Optimization
from hyperopt import hp

# Models
import sklearn
import numpy as np



def load(model):
    """
    Loads the appropriate model and hyperopt optimization space
    :param model: the name of the model to load
    :return: SKLearn model and hyperopt space
    """

    if model == 'SVC':

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

    elif model == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier

        MODEL = DecisionTreeClassifier()

        # Define the search space
        space = {
            'criterion': hp.choice('criterion',
                                   [{'criterion': 'gini'},
                                    {'criterion': 'entropy'}]),
            'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'random_state': hp.choice('random_state', [0]),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5),
            'class_weight': hp.choice('class_weight', ['balanced', None])
        }

    elif model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier

        MODEL = RandomForestClassifier()

        # Define the search space
        space = {
            'n_estimators': hp.choice('n_estimators', range(10, 300)),
            'criterion': hp.choice('criterion',
                                   [{'criterion': 'gini'},
                                    {'criterion': 'entropy'}]),
            'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'random_state': hp.choice('random_state', [0]),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5),
            'n_jobs': hp.choice('n_jobs', [8]),
            'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
        }

    elif model == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier

        MODEL = AdaBoostClassifier()

        # Define the search space
        space = {
            'n_estimators': hp.choice('n_estimators', range(10, 200)),
            'learning_rate': hp.uniform('learning_rate', 0.1, 10),
            'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
            #'random_state': hp.choice('random_state', [0]),
        }

    elif model == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingClassifier

        MODEL = GradientBoostingClassifier()

        # Define the search space
        space = {
            'loss': hp.choice('loss', ['deviance', 'exponential']),
            'learning_rate': hp.uniform('learning_rate', 0.01, 10),
            'n_estimators': hp.choice('n_estimators', range(10, 300)),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'criterion': hp.choice('criterion',
                                   [{'criterion': 'friedman_mse'},
                                    {'criterion': 'mse'},
                                    {'criterion': 'mae'}]),
            'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'max_depth': hp.choice('max_depth', range(1, 30)),
            'max_features': hp.choice('max_features', ['auto', 'log2', None]),
            'random_state': hp.choice('random_state', [0]),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5),
        }

    elif model == 'XGBoost':
        import xgboost

        MODEL = xgboost.XGBClassifier(tree_method='gpu_hist', gpu_id=0)

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

    elif model == 'MLP':
        from sklearn.neural_network import MLPClassifier

        MODEL = MLPClassifier()


        def hidden_layers(depth, max_nodes1, max_nodes2):
            options = []
            for k in range(1, max_nodes1):
                for i in range(0, depth):
                    if i == 0:
                        options.append(tuple([k, ]))
                    else:
                        for j in range(1, max_nodes2):
                            options.append(tuple([k, j]))
            return options


        layers = [(4,), (8,), (16,), (24,), (32,), (48,), (64,), (128,), (4, 2), (8, 4), (16, 8), (24, 12), (32, 16),
                  (48, 24), (64, 32), (128, 64), (128, 64, 32)]

        # Define the search space
        space = {
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', layers),
            # 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', range(2, 100, 1)),
            'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': hp.choice('solver', [{'solver': 'lbfgs'},
                                           {'solver': 'sgd',
                                            'learning_rate': hp.choice('learning_rate',
                                                                       ['constant', 'invscaling', 'adaptive'])},
                                           {'solver': 'adam'}]),
            'alpha': hp.uniform('alpha', 0, 0.001),
            'batch_size': hp.choice('batch_size', range(1, 100, 1)),
            'learning_rate_init': hp.uniform('learning_rate_init', 0.00000001, 0.1),
            'power_t': hp.uniform('power_t', 0.1, 1),
            'max_iter': hp.choice('max_iter', range(100, 300)),
            'tol': hp.uniform('tol', 0.0000001, 0.01),
            'early_stopping': hp.choice('early_stopping', [True]),
            'momentum': hp.uniform('momentum', 0.5, 0.99),
            'beta_1': hp.uniform('beta_1', 0.5, 0.9999),
            'beta_2': hp.uniform('beta_2', 0.9, 0.99999)
        }

    elif model == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression

        MODEL = LogisticRegression()

        # Define the search space
        space = {
            'solver': hp.choice('solver',
                                [{'solver': 'newton-cg',
                                  'penalty': hp.choice('newton_penalty', ['l2', 'none'])},
                                 {'solver': 'lbfgs',
                                  'penalty': hp.choice('lbfgs_penalty', ['l2', 'none'])},
                                 {'solver': 'liblinear',
                                  'penalty': hp.choice('linear_penalty', ['l1', 'l2'])},
                                 {'solver': 'sag',
                                  'penalty': hp.choice('sag_penalty', ['l2', 'none'])},
                                 {'solver': 'saga',
                                  'penalty': hp.choice('saga_penalty', ['l1', 'l2', 'elasticnet', 'none'])}]),
            'tol': hp.uniform('tol', 0.0000001, 0.1),
            'C': hp.uniform('C', 0.1, 10),
            'fit_intercept': hp.choice('fit_intercept', [True, False]),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'random_state': hp.choice('random_state', [0]),
            'max_iter': hp.choice('max_iter', range(1, 1000)),
            'multi_class': hp.choice('multi_class', ['ovr']),
            'n_jobs': hp.choice('n_jobs', [8])
        }

    else:
        print('Unknown model')
        exit

    return MODEL, space