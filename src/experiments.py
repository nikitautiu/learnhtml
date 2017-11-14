import logging

import glob
import pandas as pd
import numpy as np
import dask.dataframe as dd
import re
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate(X_train, Y_train, X_test=None, Y_test=None, model_func=None, test_size=0.3):
    """Receives a a train dataset_dragnet and optionally a test one. Does the splitting
    of the data in stratified manner so that the class proportions are preserved.
    If no test dataset_dragnet is received, splits the training one in 2."""
    if X_test is None:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=test_size, stratify=Y_train.values.argmax(1))

    scores = model_func(X_train, Y_train, X_test, Y_test)

    # the sklearn func returns a np array, convert it to a dataframe for verbosity
    scores = pd.DataFrame(data=np.array(scores).T, columns=['precision', 'recall', 'f1-score', 'support'])
    scores['label'] = Y_test.columns  # indicate the labels
    return scores


def X_Y_convert(df, label_cols=None):
    # get the label cols
    if label_cols is None:
        label_cols = df.filter(regex='[^g]_label', axis=1).columns.tolist()

    # input the training data
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')
    df['label'] = df[label_cols].idxmax(1)  # revse one-hot
    df.loc[~df[label_cols].any(1), 'label'] = 'noise'  # the irrelevant ones

    X = df.drop(['url', 'path', 'label'] + label_cols, axis='columns', errors='ignore')  # the data

    # convert to binary classes, one-hot encoded
    Y = df[label_cols]
    Y = Y.assign(noise=~Y[label_cols].any(1)).astype(float)

    return X, Y


def train_and_eval_file(train_file, test_file=None, model_func=None, label_cols=None):
    """Given a train file, test file, label to classify, and a function
    for the model returning the precision recall and fscore, return a dataframe
    of the observations."""
    X_test, Y_test = None, None  # init them this way
    df = dd.read_csv(train_file).compute()
    X_train, Y_train = X_Y_convert(df, label_cols=label_cols)

    if test_file is not None and test_file != train_file:
        # input this too, if it's different and specified
        df = dd.read_csv(test_file).compute()
        X_test, Y_test = X_Y_convert(df, label_cols=label_cols)

    return train_and_evaluate(X_train, Y_train, X_test, Y_test, model_func)


def get_dataset_descr_from_filename(filename):
    """Given a filename return the dataset_dragnet description of it.
    (website, label) meaning what website it's from and
    what labels it contains(if it contains all data from
    one website, return "all")."""
    filename = filename.split('/')[-1]  # get the fiename proper
    filename_ptrn = r'(?P<website>[\w.-]+)(-(?P<label>.*)-\*)?\.csv'
    match = re.match(filename_ptrn, filename)  # get the match obj
    return (match.group('website'), match.group('label') or 'all')  # either website only or label and website


def rf_eval(X_train, Y_train, X_test, Y_test):
    """Random forest evaluator"""
    # classify
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)

    # performance
    Y_test_pred = clf.predict(X_test)

    logging.info(classification_report(Y_test, Y_test_pred))  # for debug

    # np.array classification report
    return precision_recall_fscore_support(Y_test, Y_test_pred, warn_for=[])


# REPRODUCIBLE EXPERIMENT SECTION
# contains functions returning dataframes of classifications reports
# they are parameterizable with the model and path to se
def simple_model_experiment(experiments, model_func, experiment_name, progress=True, label_cols=None):
    """Receives a list of experiment descriptors
    each descriptor has a train csv, a test csv, a train website,
    a train label, a test website and a test label.

    It also receives the model func, the label cols and an experiment name.


    Return a dataframe of the classification reports for the experiment."""
    # get all the 1st class datasets
    test_result_dfs = []  # a list of all the test results, will concatenate at the end

    for experiment in experiments:
        # some websites do not have a label so we have to check first
        # also, check again for null files
        test_file = experiment['file_test']
        train_file = experiment['file_train']
        if len(dd.read_csv(train_file)) != 0 and len(dd.read_csv(test_file)) != 0:

            # we don't need to get the dsecription anymore, already extracted
            results = train_and_eval_file(train_file, test_file=test_file,
                                          label_cols=label_cols, model_func=model_func)

            # label them with the experiment metadata
            results['train_website'] = experiment['website_train']
            results['test_website'] = experiment['website_test']

            results['train_pages_label'] = experiment['label_train']
            results['test_pages_label'] = experiment['label_test']
            if progress:
                print('{} - {}'.format(train_file, test_file))  # for debug purposes only

            test_result_dfs.append(results)

    result_df = pd.concat(test_result_dfs, ignore_index=True)
    result_df['experiment'] = experiment_name  # add the experiment name
    return result_df
