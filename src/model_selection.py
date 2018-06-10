import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from keras_utils import create_model
from log import logger
from utils import ItemSelector, MyKerasClassifier, group_argsort, dict_combinations, MultiColumnTransformer


def get_percentile_distr():
    """Get a distribution of percentiles geometrically 
    spaced between 50-100 and 5-50. Less in the middle, more
    at the ends."""
    second_part = (np.geomspace(5, 50, 10)).astype(int)
    first_part = (101. - np.geomspace(1, 51, 20)).astype(int)
    return np.hstack([first_part, second_part])


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
    'classify__class_weight': ['balanced', None]
}]

PARAM_COMBINATIONS = {
    'logistic': [LOGISTIC_TUNABLE, MISC_TUNABLE],
    'svm': [SVM_TUNABLE, MISC_TUNABLE],
    'tree': [DECISION_TREE_TUNABLE, MISC_TUNABLE],
    'random': [RANDOM_FOREST_TUNABLE, MISC_TUNABLE],
    'deep': [MISC_TUNABLE, DEEP_TUNABLE]
}


def create_classifier(classifier_name):
    """Based on the passed name, return a Classifier object."""
    classifier_map = {
        'logistic': LogisticRegression(),
        'svm': LinearSVC(),
        'tree': DecisionTreeClassifier(),
        'random': RandomForestClassifier(),
        'deep': MyKerasClassifier(create_model, shuffle=True, expiration=2,
                                  hidden_layers=[3000, 1000, 500] + [100] * 3,
                                  optimizer='adagrad', dropout=0, activation='relu',
                                  class_weight='balanced', epochs=500, patience=100)
    }

    return classifier_map[classifier_name]


def create_pipeline(**parameters):
    """Creates a Pipeline, to use as classifier, based on the
    parameters passed. These have the same meaning as the special
    parameters passed to `get_param_grid`.

    Mainly concerned with the feature ones.
    """

    # height and depth
    height = parameters.get('height', 0)
    depth = parameters.get('depth', 0)

    # create a selector for ancestor and descendants
    height_depth_selector = create_verbosity_selectors(depth, height)

    # feature subset
    use_numeric = parameters.get('use_numeric', False)
    use_classes = parameters.get('use_classes', False)
    use_ids = parameters.get('use_ids', False)
    use_tags = parameters.get('use_tags', False)

    transformer_list = create_feature_transformers(use_classes, use_ids, use_numeric, use_tags, height, depth)

    estimator = Pipeline(steps=[
        ('verbosity', height_depth_selector),
        ('union', FeatureUnion(transformer_list=transformer_list)),
        ('normalizer', MaxAbsScaler()),
        ('reduce_dim', SelectPercentile(chi2)),
        ('classify', create_classifier(parameters.get('classify', 'logistic')))
    ])

    return estimator


def create_verbosity_selectors(depth, height):
    """Get an ItemSelector that selects features based on a height and depth"""
    # ancestor
    ancestor_regex = None
    if height != 0:
        ancestor_regex = r'(ancestor({}).+)'.format(
            r'|'.join(str(x) for x in range(1, height + 1)))
    # descendant
    depth_regex = None
    if depth != 0:
        depth_regex = r'(descendant({}).+)'.format(
            r'|'.join(str(x) for x in range(1, depth + 1)))
    regexes = [r'((?!ancestor|descendant).+)']  # generic one and the ancestor and depth

    if ancestor_regex is not None:
        regexes.append(ancestor_regex)
    if depth_regex is not None:
        regexes.append(depth_regex)

    hd_regex = r'^' + r'|'.join(regexes) + r'$'
    height_depth_selector = ItemSelector(regex=hd_regex)
    return height_depth_selector


def is_not_object(x):
    """Helper function for selecting only object cols"""
    return str(x[1]) != 'object'


def create_feature_transformers(use_classes, use_ids, use_numeric, use_tags, height, depth):
    """Get a set of transformers for the features"""

    # feature_union_creation
    transformer_list = []  # number of transformers
    if use_tags:
        # transform the tags, label binarizer for categorical tags
        # ancestor and normal
        transformer_list.append(
            ('categorical_tags', Pipeline(steps=[
                ('select', ItemSelector(regex=r'^.*tag$')),
                ('vectorize', MultiColumnTransformer(CountVectorizer(binary=True)))
            ]))
        )

        if depth > 0:
            # descendant tags. "tags" not "tag" this time
            transformer_list.append(
                ('frequency_tags', Pipeline(steps=[
                    ('select', ItemSelector(regex=r'^.*tags$')),
                    ('vectorize', MultiColumnTransformer(TfidfVectorizer(analyzer='word',
                                                                         ngram_range=(1, 1),
                                                                         use_idf=False,
                                                                         token_pattern=r'(?u)\b\w+\b')))
                ]))
            )
    if use_classes:
        # classes, use with td-idf
        transformer_list.append(
            ('classes', Pipeline(steps=[
                ('select', ItemSelector(regex=r'^(.*class_text)|.+([0-9]_classes)$')),
                ('vectorize', MultiColumnTransformer(TfidfVectorizer(analyzer='char_wb',
                                                                     ngram_range=(3, 3),
                                                                     use_idf=False)))
            ]))
        )
    if use_ids:
        # ids, use tf-idf
        transformer_list.append(
            ('ids', Pipeline(steps=[
                ('select', ItemSelector(regex=r'^(.*id_text)|(.*ids)$$')),
                ('vectorize', MultiColumnTransformer(TfidfVectorizer(analyzer='char_wb',
                                                                     ngram_range=(3, 3),
                                                                     use_idf=False)))
            ]))
        )
    if use_numeric:
        # get the numeric features
        transformer_list.append(
            ('numeric', Pipeline(steps=[
                ('select', ItemSelector(predicate=is_not_object)),
            ]))
        )
    return transformer_list


def get_param_grid(**parameters):
    """Given a set of parameters return a n estimator
    and a grid of search parameters.

    The majority of values are used for customizing the tunable parameters.
    These have the same meaning as key-values pair passed to RandomSearchCV.
    A few pairs have special meanings:
    * `classify`: Can be one of "logistic", "svm", "tree", "random" or "deep" and specifies the classifier
    * `use_numeric`: Whether to use the numeric features extracted.
    * `use_classes`: Whether to use text information from classes.
    * `use_ids`: Whether to use the id information
    * `use_tags`: Whether to use tag information.
    * `height`: How many ancestors to use.
    * `depth`: How many levels of descendants to use.

    :param parameters:
        The set of of parameters to use for initialization. Contains at least a key for
        'classify' which specifies what classifier to use.
    :type parameters: dict, contains
    """

    classifier = parameters.get('classify')  # get the classifier
    # get the other params
    pipeline_grids = PARAM_COMBINATIONS.get(classifier, None)
    if pipeline_grids is None:
        raise ValueError('classifier must be "logistic", "svm", "tree", "random" or "deep"')

    # create the estimator and parameter combinations
    estimator = create_pipeline(**parameters)

    # pop the other params
    for param in ['classify', 'use_numeric', 'use_classes', 'use_ids',
                  'use_tags', 'height', 'depth']:
        parameters.pop(param, None)

    # return the estimator and the parameter grid
    param_grid = list(dict_combinations(*pipeline_grids, [parameters]))[0]
    return estimator, param_grid


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


def cv_train(estimator, X, y, groups=None, param_distributions=None,
             n_iter=20, n_folds=5, total_folds=None,
             n_jobs=-1, scoring='f1'):
    """Train the estimator on the entire dataset(no nested CV) and return the
    resulting dataset. Parameters are the same as for `nested_cv` but
    only a set of folds is specifiable now."""

    # convert groups and labels
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(groups, pd.Series):
        groups = groups.values

    # specify the defaults
    if groups is None:
        groups = np.arange(y.shape[0])
    if total_folds is None:
        total_folds = n_folds

    # do the CV loop, pass the corresponding groups
    # we only need the best parameters
    best_params, _ = search_params(estimator, X, y, groups=groups,
                                   param_distributions=param_distributions,
                                   n_iter=n_iter, n_folds=n_folds,
                                   total_folds=total_folds, n_jobs=n_jobs,
                                   scoring=scoring)

    # refit entire dataset
    best_est = estimator
    best_est.set_params(**best_params)  # set as kwargs!
    best_est.fit(X, y)  # fit the dataset with the best params

    return best_est


def nested_cv(estimator, X, y, groups=None, param_distributions=None, n_iter=20, internal_n_folds=5,
              internal_total_folds=None, external_n_folds=5, external_total_folds=None, n_jobs=-1, scoring='f1'):
    """Perform nested cv with internal randomized CV for model selection
    Given a dataset with optional grouping, a parameter distribution for an estimator
    perform nested CV.

    The model selection is done in the internal loop which consists in a sample
    of folds from a given total(default is to use all folds - but you can basically
    use it as a split). The score is then calculated for each individual fold
    and the returned values are a list of scores and a dataframe containing all internal
    CV results.
    """
    # convert groups and labels
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(groups, pd.Series):
        groups = groups.values

    # select defaults
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
        logger.info('Model selection on fold number {}...'.format(run_nb))

        # split the dataset
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[split[0]], X.iloc[split[1]]
        else:
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
        logger.info('Refitting estimator with best params...')
        best_est = estimator
        best_est.set_params(**best_params)  # set as kwargs!
        best_est.fit(X_train, y_train)

        # add the score to the list of all scores
        scores[run_nb] = scorer(best_est, X_test, y_test)

        # log the result
        logger.info('SCORE FOR BEST ESTIMATOR ON FOLD NUMBER {} = {}'.format(run_nb, scores[run_nb]))

        # add the cross validation dataframe to the list
        cv_results['run_nb'] = run_nb
        all_cv_results.append(cv_results)

    return scores, pd.concat(all_cv_results, ignore_index=True)


def get_ordered_dataset(file_pattern, blocks_only=True, shuffle=True):
    """Given a file pattern,return the dataset contained.
    If specified, shuffle the dataset group-wise.

    :param blocks_only: whether to only use records with 'is_extracted_block' == True
    :type blocks_only: bool
    :param file_pattern: the pattern of csv from which to read the DataFrame
    :type file_pattern: str
    :param shuffle: whether to shuffle the dataset group-wise
    :type shuffle: bool
    """
    dataset = dd.read_csv(file_pattern).compute()

    is_block = np.ones(dataset.shape[0], dtype=bool)
    if blocks_only:
        # if only blocks return just the blocks
        is_block = dataset['is_extracted_block']

    data_X = dataset.drop(['content_label', 'url', 'path', 'block_text', 'is_extracted_block'], axis=1)
    data_y = dataset['content_label'][is_block]
    groups = dataset['url'][is_block]

    # fill in the missing string values
    data_X = data_X.replace(np.nan, '', regex=True)

    # order them
    order = group_argsort(groups, shuffle=shuffle)
    groups = groups.iloc[order]
    data_X = data_X.iloc[order]
    data_y = data_y.iloc[order]

    return data_X, data_y, groups
