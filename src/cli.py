#!/usr/bin/env python3
"""This module provides a utility to easily download and label website tags."""
import json
import os
import pickle
import pprint

import click
import click_log
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from dataset_conversion.conversion import convert_dataset
from features import extract_features_from_ddf
from log import logger
from model_selection import nested_cv, get_param_grid, get_ordered_dataset, cv_train

# configure the logger to use the click settings
click_log.basic_config(logger)


@click.group()
def cli():
    """Dataset creation tool"""
    pass


@cli.command(short_help='extract om features')
@click.argument('input_file', type=click.Path(file_okay=True, dir_okay=False, readable=True), metavar='INPUT_FILE')
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, readable=True), metavar='OUTPUT_DIR')
@click.option('--height', type=int, default=5, metavar='HEIGHT', help='The height of the neighbourhood')
@click.option('--depth', type=int, default=5, metavar='DEPTH', help='The depth of the neighbourhood')
@click.option('--num-workers', metavar='NUM_WORKERS', type=click.INT,
              default=8, help='The number of workers to parallelize to(default 8)')
def dom(input_file, output_dir, height, depth, num_workers):
    """Extract the dom features and output them to a directory, in a partitioned fashion"""
    dask.set_options(get=dask.multiprocessing.get, num_workers=num_workers)  # set the number of workers

    df = pd.read_csv(input_file)  # must read as pandas because dask makes a fuss about html
    feats = extract_features_from_ddf(dd.from_pandas(df, npartitions=max(num_workers * 2, 64)), depth, height)

    # output all the three to csvs
    click.echo('OUTPUTTING FEATURES')
    feats.to_csv(os.path.join(output_dir, 'feats-*.csv'), index=False)

    click.secho('DONE!', bold=True)


@cli.command(short_help='merge csv files')
@click.option('--cache', type=click.Path(dir_okay=True, file_okay=False, exists=True),
              metavar='CACHE_DIR', help='where to store cache for larger-than-memory merging',
              default=None)
@click.option('--on', type=click.STRING, metavar='MERGE_COLS', help='the columns to merge on(comma separated)')
@click.argument('output_files', metavar='OUTPUT_FILES', nargs=1)
@click.argument('input_files', metavar='INPUT_FILES', nargs=-1)
def merge(cache, output_files, input_files, on):
    """Merges the given files on the columns specified withthe --on option
    and outputs the result to output_files."""
    # set the cache if specified
    if cache is not None:
        click.echo('Using {} as cache'.format(cache))
        dask.set_options(temporary_directory=cache)

    on_columns = on.split(',')  # get the columns to merge on
    result_ddf = dd.read_csv(input_files[0])  # the first one
    for in_files in input_files[1:]:
        # merge with the others
        click.secho('MERGING {}'.format(in_files))
        in_file_ddf = dd.read_csv(in_files)
        result_ddf = result_ddf.merge(in_file_ddf, on=on_columns)

    # output it
    click.echo('OUTPUTTING')
    result_ddf.to_csv(output_files, index=False)


@cli.command(short_help='convert datasets')
@click.argument('dataset_directory', metavar='DATASET_DIRECTORY', type=click.Path(file_okay=False, dir_okay=True),
                nargs=1)
@click.argument('output_directory', metavar='OUTPUT_DIRECTORY', type=click.Path(file_okay=False, dir_okay=True),
                nargs=1)
@click.option('--raw/--no-raw', default=True, help='Whether to output the raw file')
@click.option('--labels/--no-labels', default=True, help='Whether to output the label files')
@click.option('--blocks/--no-blocks', default=True,
              help='Whether to output to the label file which tags correspond to a block')
@click.option('--num-workers', metavar='NUM_WORKERS', type=click.INT,
              default=8, help='The number of workers to parallelize to(default 8)')
@click.option('--cleaneval/--dragnet', default=False,
              help='Whether the dataset is cleaneval or dragnet(default dragnet)')
def convert(dataset_directory, output_directory, raw, labels, num_workers, cleaneval, blocks):
    """Converts the dataset from DATASET_DIRECTORY to our format and
    outputs it to OUTPUT_DIRECTORY"""
    html_ddf, label_ddf = convert_dataset(dataset_directory, 'dragnet-' if not cleaneval else 'cleaneval-',
                                          cleaneval=cleaneval, return_extracted_blocks=blocks)

    dask.set_options(get=dask.multiprocessing.get, num_workers=num_workers)  # set the number of workers
    if raw:
        # output the html
        click.echo('OUTPUTTING RAW')
        html_ddf.compute().to_csv(output_directory + '/raw.csv', index=False)
    if labels:
        # output the html
        click.echo('OUTPUTTING LABELS')
        label_ddf.compute().to_csv(output_directory + '/labels.csv', index=False)


@cli.command(short_help='train models')
@click_log.simple_verbosity_option(logger)  # add a verbosity option
@click.argument('dataset', metavar='DATASET_FILES', nargs=1)
@click.option('output', '--score-files', metavar='OUTPUT_PATTERN',
              type=click.Path(file_okay=True, dir_okay=False, writable=True),
              help='A string format for the score files. {suffix} is replaced'
                   'by "scores" and "cv" respectively.',
              default=None)
@click.option('param_file', '-j', '--param-file', metavar='PARAM_FILE',
              type=click.File(), default=None,
              help='A json file from which to read parameters')
@click.option('cli_params', '-p', '--param', type=(str, str),
              metavar='KEY VALUE',
              help='A value for a parameter given as "key value".'
                   'Values are given as json values(so quotations count).'
                   'Can be passed multiple times.',
              multiple=True)
@click.option('--external-folds', metavar='N_FOLDS TOTAL_FOLDS',
              type=click.Tuple([int, int]), default=(10, 10),
              help='The number of folds to use and the total folds '
                   'for the external loop(default 10 10). These are used for training'
                   'as well on the entire dataset.')
@click.option('--internal-folds', metavar='N_FOLDS TOTAL_FOLDS',
              type=click.Tuple([int, int]), default=(10, 10),
              help='The number of folds to use and the total folds for '
                   'the internal loop(default 10 10)')
@click.option('--n-iter', metavar='N_ITER', type=click.INT,
              default=20, help='The number of iterations for the internal '
                               'randomized search(default 20)')
@click.option('--n-jobs', metavar='N_JOBS', type=click.INT,
              default=-1, help='The number of jobs to start in parallel(default -1)')
@click.option('--random-seed', metavar='RANDOM_SEED', type=click.INT,
              default=42, help='The random seed to use')
@click.option('model_file', '--model-file', metavar='MODEL_FILE',
              type=click.Path(file_okay=True, dir_okay=False, writable=True),
              help='The file in which to save the pickled model trained'
                   'over the entire dataset.',
              default=None)
@click.option('--shuffle/--no-shuffle', default=True,
              help='Whether to shuffle the dataset beforehand')
def train(dataset, output, external_folds, internal_folds,
          n_iter, n_jobs, random_seed, param_file, model_file,
          cli_params, shuffle):
    """Trains a model over a dataset, given a set of values of parameters to use for
    the CV. Parameters used:

    """
    params = {}
    # attempt to read params from file
    if param_file is not None:
        params = json.load(param_file)

    for param in cli_params:
        # load the values from the json
        key, val = param
        loaded_val = json.loads(val)
        params[key] = loaded_val

    logger.debug('Passing params:\n{}'.format(pprint.pformat(params)))
    # extract the params
    blocks_only = params.pop('blocks_only', True)  # use only the blocks

    """Evaluate the expected f1-score with nested CV"""
    # unpacking the fold numbers
    internal_n_folds, internal_total_folds = internal_folds
    external_n_folds, external_total_folds = external_folds

    # seed the random number generator
    logger.info('Seeding the random number generator')
    np.random.seed(random_seed)
    # there is no other solution than using tf just in the worker
    # tf.set_random_seed(random_seed)

    # load the estimator
    estimator, param_distributions = get_param_grid(**params)  # get the appropriate
    logger.debug('Computed params(after default values):\n{}'.format(pprint.pformat(param_distributions)))

    # load the dataset
    logger.info('Loading the dataset')
    X, y, groups = get_ordered_dataset(dataset, blocks_only=blocks_only, shuffle=shuffle)

    # properly format params. wrap them if lists if necessary
    # rv_frozen makes an exception because it is a scipy distribution
    param_distributions = dict(
        map(lambda p: (p[0], p[1] if isinstance(p[1], list) or isinstance(p[1], rv_frozen) else [p[1]]),
            param_distributions.items()))

    # output the scores only if specified
    if output is not None:
        # training the model
        logger.info('Performing nested CV')
        scores, cv = nested_cv(estimator, X, y, groups, param_distributions=param_distributions, n_iter=n_iter,
                               internal_n_folds=internal_n_folds, internal_total_folds=internal_total_folds,
                               external_n_folds=external_n_folds, external_total_folds=external_total_folds,
                               n_jobs=n_jobs)

        # outputting
        logger.info('Saving the results')
        output_scores = output.format(suffix='scores.csv')
        output_cv = output.format(suffix='cv.csv')

        np.savetxt(output_scores, scores)
        cv.to_csv(output_cv, index=False)

    # train the model on the whole dataset only if model_file
    # is specified
    if model_file is not None:
        logger.info('Training the model over the entire dataset')
        trained_est = cv_train(estimator, X, y, groups,
                               param_distributions=param_distributions,
                               n_iter=n_iter, n_folds=external_n_folds,
                               total_folds=external_total_folds, n_jobs=n_jobs)

        logger.info('Saving the model')
        with open(model_file, 'wb') as f:
            pickle.dump(trained_est, f)  # pickle the file

    logger.info('DONE')


if __name__ == '__main__':
    cli()
