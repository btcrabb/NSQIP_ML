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
### Cloning the Repository
To download the repository, enter the following in the command line prompt:

`> git clone https://github.com/btcrabb/nsqip_pituitary`

`> cd nsqip_pituitary`

Alternatively, you can download the repository as a zip file directly from the webpage. For secuirity reasons, the raw dataset is not included in the github repository. Make a new directory called /data/raw, as shown in the project outline above and below and copy the dataset (in .csv or .xlsx format) into this folder.

    ├── data                                <- Create new data folder.
           └── raw                          <- Create new data/raw folder.
                 └── dataset.csv            <- Save the dataset here in .csv format.


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

An example from the optimization of the SVC can be seen below:

![Scores.](./reports/optimization/SVC/2021-01-17_bayes_test_SCORES.png "Balanced Accuracy Score by Iteration")

### ROC AUC Curve Comparison 

A comparison of the ROC curves, with AUC scores, for each optimized algorithm can currently be accomplished using the src/visualization/roc_auc_curves.py python script. To run this script, enter the following in the command line:

`nsqip_pituitary>python src/visualization/roc_auc_curves.py`

The results from this script, for algorithms that have currently been optimized can be seen below. Confidence intervals for the AUC scores are calculated via bootstrapping. 

![ROC Scores.](./reports/figures/ROC_AUC_comparison.png "ROC AUC comparison")

Peformance on the testing dataset is similar to performance achieved during the cross validation phase, which is shown below. This indicates that the algorithms were able to generalize well to new data and did not overfit the training data. 

![CROSSVAL ROC Scores.](./reports/figures/Crossval_ROC_AUC_comparison.png "Cross Validation ROC AUC comparison")


### Performance Characteristics Comparison (python scripts in progress)

The 2.6-BTC-roc_auc_curves.ipynb Jupyter notebook can be used to produce a table of results, with and without bootstrapped confidence intervals, to compare each optimized algorithm on the testing dataset. These tables are shown below:

![](./reports/figures/performance_characteristics.html)

The performance on the testing dataset is similar to the performance during the cross validation phase, which is shown below.

![](./reports/figures/crossval_performance_characteristics.png)

### Bar Graphs with Confidence Intervals (python scripts in progress)

For each performance characteristic, a bar graph can be generated showing the relative performance of each algorithm using the 2.6-BTC-roc_auc_curves.ipynb Jupyter notebook. A few examples are shown below:

Sensitivity                |  Specificity               
:-------------------------:|:-------------------------: 
![](./reports/figures/Sensitivity_bar_graph.png)   |  ![](./reports/figures/Specificity_bar_graph.png)    

PPV                         |    NPV
:-------------------------: | :-------------------------:
![](./reports/figures/PPV_bar_graph.png) | ![](./reports/figures/NPV_bar_graph.png)

F1 Score                    |    ROC Curve AUC
:-------------------------: | :-------------------------:
![](./reports/figures/F1-score_bar_graph.png) | ![](./reports/figures/AUC_bar_graph.png)

### Cohen's Kappa Scores for Classifier Agreement

The Jupyter notebook 2.6-BTC-roc_auc_curves.ipynb can also produce a crosstab visualization of the Cohen's Kappa coefficients for all of the classifiers. Cohen's kappa coefficient (κ) is a statistic that is used to measure inter-rater reliability for categorical items. Some fairly arbitrary guidelines in the literature identify kappas over 0.75 as excellent, 0.40 to 0.75 as fair to good, and below 0.40 as poor [Fleiss, J.L. (1981). Statistical methods for rates and proportions (2nd ed.). New York: John Wiley]. Below we see that both the linear models (SVC with linear kernel and Logistic Regression) have a high inter-rater reliability, indicating that they are producing similar predictions and likely have a similar decision boundary. 

![](./reports/figures/cohens_kappa_scores.png)

### Permutation Feature Analysis (python scripts in progress)

A permutation analysis to identify the most important features can be accomplished using the 2.5-BTC-permutations.ipynb Jupyter notebook. Shown below is the complete permutation analysis for currently included variables, from most to least important. Negative values indicate high feature importance. Positive values may indicate that the inclusion of this feature is actually hurting algorithm performance. 

![](./reports/figures/permutation_bar_graph.png)
