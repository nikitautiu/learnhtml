import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import tf_utils
from keras_utils import create_model
from utils import ItemSelector, MyKerasClassifier, dict_combinations, RecDict, group_argsort


def get_percentile_distr():
    """Get a distribution of percentiles geometrically 
    spaced between 50-100 and 5-50. Less in the middle, more
    at the ends."""
    second_part = (np.geomspace(5, 50, 10)).astype(int)
    first_part = (101. - np.geomspace(1, 51, 20)).astype(int)
    return np.hstack([first_part, second_part])


# define the common pipeline for all the model selection
PIPELINE_EST = Pipeline(steps=[
    ('union', FeatureUnion(transformer_list=[
        ('numeric', Pipeline(steps=[
            ('select', ItemSelector(key='numeric')),
        ])),
        ('class', Pipeline(steps=[
            ('select', ItemSelector(key='text')),
            ('text', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), use_idf=False))
        ]))],
        transformer_weights={
            'numeric': 1.0,
            'class': 1.0
        },
    )),
    ('normalizer', MaxAbsScaler()),
    ('reduce_dim', SelectPercentile(chi2)),
    ('classify', None)
])

# define all the fixed params for each of the models

LOGISTIC_FIXED = [{
    'classify': [LogisticRegression()]
}]

SVM_FIXED = [{
    'classify': [LinearSVC()]
}]

DECISION_TREE_FIXED = [{
    'classify': [DecisionTreeClassifier()]
}]

RANDOM_FOREST_FIXED = [{
    'classify': [RandomForestClassifier()]
}]

DEEP_FIXED = [{
    'classify': [MyKerasClassifier(create_model, shuffle=True, expiration=2,
                                   hidden_layers=[3000, 1000, 500] + [100] * 3,
                                   optimizer='adagrad', dropout=0, activation='relu',
                                   class_weight='balanced', epochs=500, patience=100)],
}]

# define all the tunable params for each of them

LOGISTIC_TUNABLE = [{
    'classify__penalty': ['l1', 'l2'],  # l1 and l2 regularization, l1 introduces sparsity(lasso)
    'classify__C': stats.reciprocal(a=1e-1, b=1e4)
}]

SVM_TUNABLE = [{
    'classify__penalty': ['l2'],
    'classify__C': stats.reciprocal(a=1e-1, b=1e4),
}]

DECISION_TREE_TUNABLE = [{
    'classify__max_features': ['sqrt', 'log2']
}]

RANDOM_FOREST_TUNABLE = [{
    'classify__max_features': ['sqrt', 'log2']
}]

DEEP_TUNABLE = [{
    'classify__optimizer': ['adagrad', 'adam', 'rmsprop'],
    'classify__activation': ['relu', 'selu', 'sigmoid', 'tanh'],
    'classify__dropout': stats.uniform(0., .4),
    'classify__hidden_layers': [
        [1000],
        [1000, 500],
        [1000, 500, 100],
        [1000, 500, 100, 100],
    ]
}]

MISC_TUNABLE = [{
    'reduce_dim__percentile': get_percentile_distr(),
    'union__class__text__use_idf': [True, False],
    'union__class__text__analyzer': ['char_wb', 'word'],
    'union__class__text__ngram_range': [(1, 1), (3, 3)],
    'classify__class_weight': ['balanced', None]
}]

PARAM_COMBINATIONS = {
    'logistic': [LOGISTIC_FIXED, LOGISTIC_TUNABLE, MISC_TUNABLE],
    'svm': [SVM_FIXED, SVM_TUNABLE, MISC_TUNABLE],
    'tree': [DECISION_TREE_FIXED, DECISION_TREE_TUNABLE, MISC_TUNABLE],
    'random': [RANDOM_FOREST_FIXED, RANDOM_FOREST_TUNABLE, MISC_TUNABLE],
    'deep': [MISC_TUNABLE, DEEP_FIXED, DEEP_TUNABLE]
}


def get_param_grid(classifier, features):
    """Given a classifier name and the feature set name,
    return its corresponding estimator and param search grid"""

    # compute the feature weights
    text_weight = 0.
    numeric_weight = 0.
    if features == 'numeric':
        numeric_weight = 1.
    elif features == 'text':
        text_weight = 1.
    elif features == 'both':
        numeric_weight = 1.
        text_weight = 1.
    else:
        raise ValueError('features must be "text", "numeric" or "both')

    feature_weight_grid = [{
        'union__transformer_weights': [
            {'numeric': numeric_weight, 'class': text_weight}
        ]
    }]

    # get the other params
    pipeline_grids = PARAM_COMBINATIONS.get(classifier, None)
    if pipeline_grids is None:
        raise ValueError('classifier must be "logistic", "svm", "tree", "random" or "deep"')

    return PIPELINE_EST, list(dict_combinations(feature_weight_grid, *pipeline_grids))[0]


def generate_grouped_splits(X, y, groups, total_folds=10, n_folds=10):
    """Generate n grouped folds from a total of total_folds"""
    k_fold_splitter = GroupKFold(total_folds)
    return list(k_fold_splitter.split(X, y, groups))[:n_folds]


def search_params(estimator, X, y, groups=None, param_distributions=None,
                  n_iter=20, n_folds=5, total_folds=None, n_jobs=-1, scoring='f1'):
    """Given and estimator, and some parameter distributions, do a randomized search
    of n_iter on the specified number of folds.

     In the end, return the best estimator and the dataframe of the cv results"""
    if groups is None:
        groups = np.arange(y.shape[0])
    if total_folds is None:
        # if total folds is not specified assume the same number
        # otherwise we would only use a portion of them
        total_folds = n_folds

    # predefine the splits
    splits = generate_grouped_splits(X, y, groups,
                                     total_folds=total_folds, n_folds=n_folds)

    # fit the gridsearch
    # but do not use refit as it will prompt cloning of the estimators
    # which leads to a memory leak with Keras
    searcher = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions,
                                  n_iter=n_iter, refit=False, scoring=scoring, return_train_score=True,
                                  cv=splits, verbose=2, n_jobs=n_jobs, pre_dispatch='2+n_jobs')
    searcher.fit(X, y, groups=groups)

    return searcher.best_params_, pd.DataFrame(searcher.cv_results_)


def nested_cv(estimator, X, y, groups=None, param_distributions=None,
              n_iter=20, internal_n_folds=5, internal_total_folds=None,
              external_n_folds=5, external_total_folds=None,
              n_jobs=-1, scoring='f1', verbose=True):
    """Perform nested cv with internal randomized CV for model selection
    Given a dataset with optional grouping, a parameter distribution for an estimator
    perform nested CV.

    The model selection is done in the internal loop which consists in a sample
    of folds from a given total(default is to use all folds - but you can basically
    use it as a split). The score is then calculated for each individual fold
    and the returned values are a list of scores and a dataframe containing all internal
    CV results.
    """
    if groups is None:
        groups = np.arange(y.shape[0])
    if internal_total_folds is None:
        internal_total_folds = internal_n_folds
    if external_total_folds is None:
        external_total_folds = external_n_folds

    # get the external splits
    splits = generate_grouped_splits(X, y, groups, total_folds=external_total_folds,
                                     n_folds=external_n_folds)

    # list in which to store all cv results
    all_cv_results = []

    # get the scorer class for the metrics
    scorer = get_scorer(scoring)
    scores = np.zeros(external_n_folds, dtype='float32')

    for run_nb, split in zip(range(external_n_folds), splits):
        if verbose:
            # monitor message
            print('Model selection on fold number {}...'.format(run_nb))

        # split the dataset
        X_train, X_test = X[split[0]], X[split[1]]
        y_train, y_test = y[split[0]], y[split[1]]
        groups_train, groups_test = groups[split[0]], groups[split[1]]

        # do the internal loop, pass the corresponding groups
        best_params, cv_results = search_params(estimator, X_train, y_train, groups=groups_train,
                                                param_distributions=param_distributions,
                                                n_iter=n_iter, n_folds=internal_n_folds,
                                                total_folds=internal_total_folds, n_jobs=n_jobs,
                                                scoring=scoring)

        # refit the the best estimator with all the data
        if verbose:
            print('Refitting estimator with best params...')
        best_est = estimator
        best_est.set_params(**best_params)  # set as kwargs!
        best_est.fit(X_train, y_train)

        # add the score to the list of all scores
        scores[run_nb] = scorer(best_est, X_test, y_test)

        # log the result
        if verbose:
            print('SCORE FOR BEST ESTIMATOR ON FOLD NUMBER {} = {}'.format(run_nb, scores[run_nb]))

        # add the cross validation dataframe to the list
        cv_results['run_nb'] = run_nb
        all_cv_results.append(cv_results)

    return scores, pd.concat(all_cv_results, ignore_index=True)


def get_ordered_dataset(file_pattern, blocks_only=True, shuffle=True, block_text=False):
    """Given a file pattern,return the dataset contained.
    If specified, shuffle the dataset group-wise."""
    block_cols = ['block_text'] if block_text else []
    dataset = tf_utils.get_numpy_dataset(file_pattern, text_cols=['class_text', 'id_text'] + block_cols)

    is_block = np.ones(dataset['y'].shape[0], dtype=bool)
    if blocks_only:
        # if only blocks return just the blocks
        is_block = dataset['is_block'].ravel()

    data_X = RecDict({'numeric': dataset['numeric'][is_block], 'text': dataset['text'][is_block, 0]})
    data_y = dataset['y'][is_block]
    groups = dataset['id'][is_block]

    # order them
    order = group_argsort(groups, shuffle=shuffle)
    groups = groups[order]
    data_X = data_X[order]
    data_y = data_y[order]

    return data_X, data_y, groups