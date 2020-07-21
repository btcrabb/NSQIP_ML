NSQIP_pituitary
==============================

Predictive algorithms for NSQIP pituitary tumors

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Getting Started

### Installing necessary packages

All of the necessary python packages can be installed using either conda or pip. To install necessary packages with pip, enter the following in the command line:

`> pip3 install -r requirements.txt` (Python 3)

To install the necessary packages using Conda, use the following commands to create a Conda virtual environment with all of the necessary packages:

`> conda env create -f environment.yml`

The environment will be name NSQIP, and you can activate it by typing:

`> conda activate NSQIP`

### Generating Clean Dataset

After all of the necessary packages are installed, we can generate a cleaned dataset from the raw data. Options for this process can be found in src/dataset_options.py. Using this file or the command line, you can select which features you would like to include in the cleaned dataset (e.g., AGE, SEX, WTLOSS, etc). You can also select how to handle missing values, how to normalize continuous variables, and the size of the testing dataset. From the main project directory, run the following:

`nsqip_pituitary>python src/data/make_dataset.py --input_file data/raw/<insert filename>`

To see a complete list of available options, run:

`nsqip_pituitary>python src/data/make_dataset.py --help`

By default, a complete dataset will be generated in data/processed/NSQIP_processed.csv. Furthermore, datasets containing the training features, training labels, testing features, and testing labels will be saved in data/split/ as follows:

    ├── data
        │   ├── external       <- Data from third party sources.
        │   ├── interim        <- Intermediate data that has been transformed.
        │   ├── processed      <- The final, canonical data sets for modeling.
        │   ├── raw            <- The original, immutable data dump.
        |   └── split          <- The dataset split into training and testing datasets.
                |    ├── test_features.csv
                |    ├── test_labels.csv
                |    ├── train_features.csv
                |    └── train_labels.csv
             
### Running Bayesian Parameter Optimization

Now that the datasets have been created, we can tune the parameters of any SKLearn algorithm (such as SVC, MLP, XGBoost, RandomForest, DecisionTree, etc) using src/models/optimization.py. All options for this script can be found in src/options/optimization_options.py and can be changed via the command line or directly in the file.
To run the optimization for the Suppert Vector Classifier (SVC) for 1000 iterations, enter:

`nsqip_pituitary>python src/models/optimization.py --model SVC --max_evals 1000`

The output will look something like the following:

    ----------------- Options ---------------
              checkpoints_dir: ./checkpoints
                 continue_opt: 0
                 dataset_path: data/split/
                      gpu_ids: -1
                    max_evals: 10                                   [default: 5000]
                       metric: balanced_accuracy_score
                        model: SVC
                      n_folds: 5
                         name: experiment_name
                  num_threads: 4
                      outfile: reports/optimization/2020-07-20_bayes_test.csv
    ----------------- End -------------------
    100%|███████████████████████████████████████████████| 10/10 [00:18<00:00,  1.81s/trial, best loss: 0.40362455962455956]
    The highest cross validation score from Bayesian was 0.59638 found on iteration 3.
    Optimal Hyperparameters: {'C': 5.554313452624174, 'class_weight': 'balanced', 'kernel': 'linear', 'tol': 0.0023771944076088633}
    ROC AUC from Bayesian on test data = 0.73037.
    Optimization Complete

This script will record each trial and save all results to reports/optimization/ by default. The optimization can be continued for additional tuning if desired using the --continue_opt = 1 command line flag. 

Multiple algorithms can be optimized sequentially using the file src/models/optimize_multiple.py. The algorithms to optimize are specified at the start of this script and can be edited manually. 

### Visualizing Optimization Results

The results of the Bayesian parameter optimization can be visualized using the script src/visualization/visualize_optimization.py. This scrip will plot the value of each parameter optimized per iteration so you can see how it changes over time. To visualize our SVC optimization from the previous step, enter:

`nsqip_pituitary>python src/visualization/visualize_optimization.py --results reports/optimization/<enter today's date in YYYY-MM-DD format>_bayes_test.csv`

An example from the optimization of a multi-layered perceptron (MLP) can be seen below:

![Learning Rate.](./reports/optimization/MLP/learning_rate_init_by_iteration.png "Learning Rate by Iteration")
![Momentum.](./reports/optimization/MLP/momentum_by_iteration.png "Momentum by Iteration")
![Scores.](./reports/optimization/MLP/2020-07-16_bayes_test_SCORES.png "Balanced Accuracy Score by Iteration")

### ROC AUC Curve Comparison (python scripts in progress)

A comparison of the ROC curves, with AUC scores, for each optimized algorithm can currently be accomplished using the 2.6-BTC-roc_auc_curves.ipynb Jupyter notebook. The results from this notebook, for algorithms that have currently been optimized can be seen below. Confidence intervals for the AUC scores are calculated via bootstrapping. 

![ROC Scores.](./reports/figures/ROC_AUC_comparison.png "ROC AUC comparison")

Currently, the best performance was achieved by the SVC, LogisticRegression, and Random Forest algorithms. Additional performance metrics for these algorithms are shown below:

    LogisticRegression(C=6.8786446082731025, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=120, multi_class='ovr', n_jobs=8, penalty='none',
                       random_state=0, solver='sag', tol=0.002870818059540498,
                       verbose=0, warm_start=False)
                  precision    recall  f1-score   support

               0       0.94      0.74      0.83       270
               1       0.20      0.60      0.31        30

        accuracy                           0.73       300
       macro avg       0.57      0.67      0.57       300
    weighted avg       0.87      0.73      0.78       300

    Confusion Matrix:
    Predicted   0    1
    Actual            
    0          200  70
    1           12  18

    Sensitivity: 0.6
    Specificity: 0.7407407407407407
    PPV: 0.20454545454545456
    NPV: 0.9433962264150944

    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                           class_weight='balanced_subsample', criterion='gini',
                           max_depth=None, max_features='log2', max_leaf_nodes=None,
                           max_samples=None,
                           min_impurity_decrease=0.008135280338330211,
                           min_impurity_split=None, min_samples_leaf=6,
                           min_samples_split=6,
                           min_weight_fraction_leaf=0.04357678838454903,
                           n_estimators=253, n_jobs=8, oob_score=False,
                           random_state=0, verbose=0, warm_start=False)
                  precision    recall  f1-score   support

               0       0.93      0.87      0.90       270
               1       0.25      0.40      0.31        30

        accuracy                           0.82       300
       macro avg       0.59      0.63      0.60       300
    weighted avg       0.86      0.82      0.84       300

    Confusion Matrix:
    Predicted   0    1
    Actual            
    0          234  36
    1           18  12

    Sensitivity: 0.4
    Specificity: 0.8666666666666667
    PPV: 0.25
    NPV: 0.9285714285714286

    SVC(C=0.13869718045614998, break_ties=False, cache_size=200,
        class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3,
        gamma='scale', kernel='linear', max_iter=-1, probability=True,
        random_state=None, shrinking=True, tol=0.003993171050079279, verbose=False)
                  precision    recall  f1-score   support

               0       0.95      0.80      0.87       270
               1       0.26      0.63      0.37        30

        accuracy                           0.78       300
       macro avg       0.61      0.72      0.62       300
    weighted avg       0.88      0.78      0.82       300

    Confusion Matrix:
    Predicted   0    1
    Actual            
    0          216  54
    1           11  19

    Sensitivity: 0.6333333333333333
    Specificity: 0.8
    PPV: 0.2602739726027397
    NPV: 0.9515418502202643

#### Notes on Linear Classifiers:

It's important to highlight here that the support vector machine classifier (SVC) is using a linear kernel as the basis function. It is unsuprising that logistic regression performs well also, since logistic regression is a linear model as well. Through the bayesian optimization process, the SVC with a linear kernel outperformed all other available kernel options (‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’). It is possible for both logistic regression and the linear SVC to produce similar decision boundaries (thus similar performance) in this scenario. 

### Performance Characteristics Comparison (python scripts in progress)

The 2.6-BTC-roc_auc_curves.ipynb Jupyter notebook can be used to produce a table of results, with and without bootstrapped confidence intervals, to compare each optimized algorithm. These tables are shown below:

<style  type="text/css" >
    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0   {
          margin: 0;
          font-family: "Helvetica", "Arial", sans-serif;
          border-collapse: collapse;
          border: none;
          column-gap: 400px;
    }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0 tbody tr:nth-child(even) {
          background-color: #fff;
    }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0 tbody tr:nth-child(odd) {
          background-color: #eee;
    }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0 td {
          padding: .4em;
    }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0 th {
          font-size: 100%;
          text-align: center;
    }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col1 {
            font-weight:  bold;
        }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col0 {
            font-weight:  bold;
        }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col5 {
            font-weight:  bold;
        }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col2 {
            font-weight:  bold;
        }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col3 {
            font-weight:  bold;
        }    #T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col4 {
            font-weight:  bold;
        }</style><table id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Sensitivity</th>        <th class="col_heading level0 col1" >Specificity</th>        <th class="col_heading level0 col2" >PPV</th>        <th class="col_heading level0 col3" >NPV</th>        <th class="col_heading level0 col4" >F1-score</th>        <th class="col_heading level0 col5" >AUC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row0" class="row_heading level0 row0" >AdaBoost</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col0" class="data row0 col0" >0.067</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col1" class="data row0 col1" >0.952</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col2" class="data row0 col2" >0.133</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col3" class="data row0 col3" >0.902</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col4" class="data row0 col4" >0.089</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row0_col5" class="data row0 col5" >0.594</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row1" class="row_heading level0 row1" >DecisionTree</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row1_col0" class="data row1 col0" >0.367</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row1_col1" class="data row1 col1" >0.789</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row1_col2" class="data row1 col2" >0.162</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row1_col3" class="data row1 col3" >0.918</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row1_col4" class="data row1 col4" >0.224</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row1_col5" class="data row1 col5" >0.642</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row2" class="row_heading level0 row2" >GradientBoosting</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col0" class="data row2 col0" >0.667</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col1" class="data row2 col1" >0.596</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col2" class="data row2 col2" >0.155</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col3" class="data row2 col3" >0.942</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col4" class="data row2 col4" >0.252</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row2_col5" class="data row2 col5" >0.631</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row3" class="row_heading level0 row3" >LogisticRegression</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col0" class="data row3 col0" >0.600</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col1" class="data row3 col1" >0.741</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col2" class="data row3 col2" >0.205</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col3" class="data row3 col3" >0.943</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col4" class="data row3 col4" >0.305</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row3_col5" class="data row3 col5" >0.741</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row4" class="row_heading level0 row4" >MLP</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row4_col0" class="data row4 col0" >0.167</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row4_col1" class="data row4 col1" >0.922</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row4_col2" class="data row4 col2" >0.192</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row4_col3" class="data row4 col3" >0.909</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row4_col4" class="data row4 col4" >0.179</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row4_col5" class="data row4 col5" >0.583</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row5" class="row_heading level0 row5" >RandomForest</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row5_col0" class="data row5 col0" >0.400</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row5_col1" class="data row5 col1" >0.867</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row5_col2" class="data row5 col2" >0.250</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row5_col3" class="data row5 col3" >0.929</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row5_col4" class="data row5 col4" >0.308</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row5_col5" class="data row5 col5" >0.721</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row6" class="row_heading level0 row6" >SVC</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col0" class="data row6 col0" >0.633</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col1" class="data row6 col1" >0.800</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col2" class="data row6 col2" >0.260</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col3" class="data row6 col3" >0.952</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col4" class="data row6 col4" >0.369</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row6_col5" class="data row6 col5" >0.736</td>
            </tr>
            <tr>
                        <th id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0level0_row7" class="row_heading level0 row7" >XGBoost</th>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row7_col0" class="data row7 col0" >0.400</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row7_col1" class="data row7 col1" >0.796</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row7_col2" class="data row7 col2" >0.179</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row7_col3" class="data row7 col3" >0.923</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row7_col4" class="data row7 col4" >0.247</td>
                        <td id="T_b4de0b98_cb2d_11ea_b3a5_00dbdfd5fbf0row7_col5" class="data row7 col5" >0.655</td>
            </tr>
    </tbody></table><br><br><style  type="text/css" >
    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0   {
          margin: 0;
          font-family: "Helvetica", "Arial", sans-serif;
          border-collapse: collapse;
          border: none;
          column-gap: 400px;
    }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0 tbody tr:nth-child(even) {
          background-color: #fff;
    }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0 tbody tr:nth-child(odd) {
          background-color: #eee;
    }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0 td {
          padding: .4em;
    }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0 th {
          font-size: 100%;
          text-align: center;
    }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col1 {
            font-weight:  bold;
        }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col0 {
            font-weight:  bold;
        }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col5 {
            font-weight:  bold;
        }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col2 {
            font-weight:  bold;
        }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col3 {
            font-weight:  bold;
        }    #T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col4 {
            font-weight:  bold;
        }</style><table id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Sensitivity</th>        <th class="col_heading level0 col1" >Specificity</th>        <th class="col_heading level0 col2" >PPV</th>        <th class="col_heading level0 col3" >NPV</th>        <th class="col_heading level0 col4" >F1-score</th>        <th class="col_heading level0 col5" >AUC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row0" class="row_heading level0 row0" >AdaBoost</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col0" class="data row0 col0" >0.07 (0.00 to 0.17)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col1" class="data row0 col1" >0.95 (0.93 to 0.97)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col2" class="data row0 col2" >0.13 (0.00 to 0.35)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col3" class="data row0 col3" >0.90 (0.86 to 0.94)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col4" class="data row0 col4" >0.09 (0.00 to 0.22)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row0_col5" class="data row0 col5" >0.59 (0.51 to 0.69)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row1" class="row_heading level0 row1" >DecisionTree</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row1_col0" class="data row1 col0" >0.37 (0.20 to 0.55)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row1_col1" class="data row1 col1" >0.79 (0.74 to 0.84)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row1_col2" class="data row1 col2" >0.16 (0.08 to 0.25)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row1_col3" class="data row1 col3" >0.92 (0.88 to 0.95)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row1_col4" class="data row1 col4" >0.22 (0.12 to 0.34)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row1_col5" class="data row1 col5" >0.64 (0.54 to 0.75)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row2" class="row_heading level0 row2" >GradientBoosting</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col0" class="data row2 col0" >0.67 (0.48 to 0.83)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col1" class="data row2 col1" >0.60 (0.54 to 0.65)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col2" class="data row2 col2" >0.16 (0.09 to 0.22)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col3" class="data row2 col3" >0.94 (0.91 to 0.98)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col4" class="data row2 col4" >0.25 (0.16 to 0.35)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row2_col5" class="data row2 col5" >0.63 (0.54 to 0.72)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row3" class="row_heading level0 row3" >LogisticRegression</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col0" class="data row3 col0" >0.60 (0.42 to 0.78)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col1" class="data row3 col1" >0.74 (0.69 to 0.80)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col2" class="data row3 col2" >0.20 (0.12 to 0.29)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col3" class="data row3 col3" >0.94 (0.91 to 0.97)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col4" class="data row3 col4" >0.31 (0.20 to 0.41)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row3_col5" class="data row3 col5" >0.74 (0.63 to 0.85)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row4" class="row_heading level0 row4" >MLP</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row4_col0" class="data row4 col0" >0.17 (0.04 to 0.31)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row4_col1" class="data row4 col1" >0.92 (0.89 to 0.95)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row4_col2" class="data row4 col2" >0.19 (0.04 to 0.36)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row4_col3" class="data row4 col3" >0.91 (0.87 to 0.94)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row4_col4" class="data row4 col4" >0.18 (0.04 to 0.31)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row4_col5" class="data row4 col5" >0.58 (0.48 to 0.65)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row5" class="row_heading level0 row5" >RandomForest</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row5_col0" class="data row5 col0" >0.40 (0.22 to 0.58)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row5_col1" class="data row5 col1" >0.87 (0.83 to 0.91)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row5_col2" class="data row5 col2" >0.25 (0.13 to 0.38)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row5_col3" class="data row5 col3" >0.93 (0.89 to 0.96)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row5_col4" class="data row5 col4" >0.31 (0.17 to 0.44)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row5_col5" class="data row5 col5" >0.72 (0.64 to 0.81)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row6" class="row_heading level0 row6" >SVC</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col0" class="data row6 col0" >0.63 (0.47 to 0.82)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col1" class="data row6 col1" >0.80 (0.75 to 0.85)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col2" class="data row6 col2" >0.26 (0.17 to 0.38)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col3" class="data row6 col3" >0.95 (0.92 to 0.98)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col4" class="data row6 col4" >0.37 (0.25 to 0.49)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row6_col5" class="data row6 col5" >0.74 (0.63 to 0.84)</td>
            </tr>
            <tr>
                        <th id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0level0_row7" class="row_heading level0 row7" >XGBoost</th>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row7_col0" class="data row7 col0" >0.40 (0.23 to 0.59)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row7_col1" class="data row7 col1" >0.80 (0.75 to 0.85)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row7_col2" class="data row7 col2" >0.18 (0.09 to 0.28)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row7_col3" class="data row7 col3" >0.92 (0.89 to 0.96)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row7_col4" class="data row7 col4" >0.25 (0.13 to 0.37)</td>
                        <td id="T_b4c5d91a_cb2d_11ea_acfc_00dbdfd5fbf0row7_col5" class="data row7 col5" >0.65 (0.55 to 0.76)</td>
            </tr>
    </tbody></table>
    



### Permutation Feature Analysis (python scripts in progress)

A permutation analysis to identify the most important features can be accomplished using the 2.5-BTC-permutations.ipynb Jupyter notebook. Shown below is the complete permutation analysis for currently included variables, from most to least important. Negative values indicate high feature importance. Positive values may indicate that the inclusion of this feature is actually hurting algorithm performance. 

    RETURNOR_Yes                                             -0.384822
    PRSODM                                                   -0.281458
    BMI                                                      -0.134625
    PRPTT                                                    -0.134017
    TOTHLOS                                                  -0.128810
    PRBUN                                                    -0.096114
    PRCREAT                                                  -0.087605
    PRPLATE                                                  -0.082976
    OPTIME                                                   -0.063349
    AGE                                                      -0.037792
    PRALBUM                                                  -0.037310
    PRINR                                                    -0.030967
    RACE_NEW_White                                           -0.026444
    ASACLAS_3_Severe_Disturb                                 -0.023277
    HYPERMED_Yes                                             -0.020323
    SEX_male                                                 -0.019657
    ASACLAS_2_Mild_Disturb                                   -0.018632
    SMOKE_Yes                                                -0.018173
    RACE_NEW_Black_or_African_American                       -0.016183
    DISCHDEST_Home                                           -0.012090
    PRPT                                                     -0.010284
    STEROID_Yes                                              -0.009741
    DIABETES_NON_INSULIN                                     -0.007751
    BLEEDIS_No                                               -0.006718
    DISCHDEST_Unknown                                        -0.006274
    WTLOSS_Yes                                               -0.005823
    BLEEDIS_Unknown                                          -0.005578
    RACE_NEW_Unknown_Not_Reported                            -0.004966
    DYSPNEA_No                                               -0.004713
    ASACLAS_Unknown                                          -0.003581
    ASACLAS_None_assigned                                    -0.003390
    ETHNICITY_HISPANIC_N                                     -0.003153
    ETHNICITY_HISPANIC_U                                     -0.002885
    DIABETES_NO                                              -0.002288
    ASACLAS_1_No_Disturb                                     -0.002257
    DYSPNEA_MODERATE_EXERTION                                -0.001813
    DIABETES_INSULIN                                         -0.001385
    RACE_NEW_American_Indian_or_Alaska_Native                -0.001377
    TRANST_Not_transferred__admitted_from_home_              -0.001377
    TRANST_Transfer_from_other                               -0.000819
    TRANST_Outside_emergency_department                      -0.000773
    BLEEDIS_Yes                                              -0.000742
    EMERGNCY_Yes                                             -0.000620
    PRSEPIS_None                                             -0.000566
    DISCHDEST_Rehab                                          -0.000214
    RACE_NEW_Native_Hawaiian_or_Pacific_Islander             -0.000145
    DYSPNEA_AT_REST                                          -0.000061
    TRANST_Unknown                                           -0.000046
    PRSEPIS_Sepsis                                           -0.000015
    DISCANCR_Yes                                             -0.000008
    WNDINF_Yes                                                0.000000
    TRANST_From_acute_care_hospital_inpatient                 0.000145
    ASACLAS_4_Life_Threat                                     0.000344
    PRSEPIS_SIRS                                              0.000413
    HXCOPD_Yes                                                0.000543
    ETHNICITY_HISPANIC_Y                                      0.000765
    TRANST_Nursing_home___Chronic_care___Intermediate_care    0.001553
    DISCHDEST_Skilled_Care__Not_Home                          0.003543
    RACE_NEW_Asian                                            0.004813
