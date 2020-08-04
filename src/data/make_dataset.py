# -*- coding: utf-8 -*-
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
from pathlib import Path
from util import util
from options.dataset_options import DatasetOptions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    opt = DatasetOptions().parse()

    # load the dataset and select features
    dataset = pd.read_csv(opt.input_file)
    data = dataset[opt.feature_list]

    # drop rows that don't have opt.label defined
    data = data[data[opt.label].notna()]
	
	# define label dataframe
    label = pd.DataFrame(data[opt.label])

	# drop the label from main dataset
    data = data.drop(columns=[opt.label])

    # select categorical features
    objects = data.select_dtypes(include='object')

    if 'RACE_NEW' in opt.feature_list:
        objects['RACE_NEW'] = objects['RACE_NEW'].fillna('Unknown/Not Reported')
    if 'ETHNICITY_HISPANIC' in opt.feature_list:
        objects['ETHNICITY_HISPANIC'] = objects['ETHNICITY_HISPANIC'].fillna('U')

    if not opt.no_encode_unknown:
        objects = objects.fillna('Unknown')
    # pandas get dummies - convert to one_hot_encoding
    for column in objects.columns:
        if len(objects.groupby([column]).size()) > 2:
                new = pd.get_dummies(objects[column], prefix=column)
                objects = pd.concat([objects, new], axis=1)
                objects = objects.drop(columns=column)
        else:
            new = pd.get_dummies(objects[column], prefix=column, drop_first=True)
            objects = pd.concat([objects, new], axis=1)
            objects = objects.drop(columns=column)

    # Change categorical encodings
    labelencoder = preprocessing.LabelEncoder()
    for column in objects.columns:
        objects[column] = labelencoder.fit_transform(objects[column].astype(str))

    # perform label encoding for target label
    label = labelencoder.fit_transform(label.astype(str))

    # select numerical features
    numerical = data.select_dtypes(include=np.number).copy()

    # replace nan values with median or mean
    for col in numerical.columns:
        if opt.fill_missing == 'median':
            numerical[col].replace({-99.0: np.NaN}, inplace=True)
            numerical[col].fillna(numerical[col].median(), inplace=True)
        elif opt.fill_missing == 'mean':
            numerical[col].replace({-99.0: np.NaN}, inplace=True)
            numerical[col].fillna(numerical[col].mean(), inplace=True)
        else:
            print('{} not a valid option for fill_missing [use mean or median]'.format(opt.fill_missing))

    # replace height and weight with BMI value
    if not opt.no_bmi:
        def calculate_bmi(weight, height):
            return (weight / (height ** 2)) * 703

        bmi_list = []
        for index, row in numerical[['WEIGHT', 'HEIGHT']].iterrows():
            bmi = calculate_bmi(row['WEIGHT'], row['HEIGHT'])
            bmi_list.append(bmi)

        numerical['BMI'] = bmi_list
        numerical = numerical.drop(columns=['WEIGHT', 'HEIGHT'])

    # normalize the scalar values
    if not opt.no_normalize:
        scaler = preprocessing.StandardScaler().fit(numerical)
        stand_dataset = scaler.transform(numerical)
        stand_dataset = pd.DataFrame(stand_dataset, index=numerical.index, columns=numerical.columns)

    # recombine categorical and numerical dataframes
    final_df = pd.concat([objects, stand_dataset], axis=1)
    final_df[opt.label] = label
	
	# check if data directory exists, create if not:
    util.mkdir(os.path.dirname(opt.output_file))
    util.mkdir(os.path.dirname(opt.train_test_datasets_path))
        
    # save to csv
    final_df.to_csv(opt.output_file)

    if not opt.no_train_test_split:

        # Extract the labels
        labels = np.array(final_df[opt.label].astype(np.int32)).reshape((-1,))
        features = final_df.drop(columns=[opt.label])
        features.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in features.columns]

        # Split into training and testing data
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=opt.testing_size,
                                                                                    random_state=0)

        print('Train shape: ', train_features.shape)
        print('Test shape: ', test_features.shape)

        print('Saving testing and training datasets')
        # Save as individual .csv files
        pd.DataFrame(train_features).to_csv(opt.train_test_datasets_path + 'train_features.csv')
        pd.DataFrame(test_features).to_csv(opt.train_test_datasets_path + 'test_features.csv')
        pd.DataFrame(train_labels).to_csv(opt.train_test_datasets_path + 'train_labels.csv')
        pd.DataFrame(test_labels).to_csv(opt.train_test_datasets_path + 'test_labels.csv')

        print('Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()