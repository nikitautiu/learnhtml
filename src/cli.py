#!/usr/bin/env python3
"""This module provides a utility to easily download and label website tags."""

import json
import os
import pickle
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

import numpy as np
import click
import dask
import dask.dataframe as dd
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from dataset_conversion.conversion import convert_dataset
from features import extract_features_from_ddf
from model_selection import nested_cv, get_param_grid, get_ordered_dataset


@click.group()
def cli():
    """Dataset creation tool"""
    pass

@cli.command()
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
    oh, freqs, feats = extract_features_from_ddf(dd.from_pandas(df, npartitions=max(num_workers * 2, 64)), depth,
                                                 height)

    # output all the three to csvs
    click.echo('OUTPUTING FEATURES')
    feats.to_csv(os.path.join(output_dir, 'feats-*.csv'), index=False)

    click.echo('OUTPUTING ONE-HOT')
    oh.to_csv(os.path.join(output_dir, 'oh-*.csv'), index=False)

    click.echo('OUTPUTING FREQUENCIES')
    freqs.to_csv(os.path.join(output_dir, 'freqs-*.csv'), index=False)

    click.secho('DONE!', bold=True)


@cli.command()
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


@cli.command()
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


@cli.command()
@click.argument('dataset', metavar='DATASET_FILES', nargs=1)
@click.argument('output', metavar='OUTPUT',
                type=click.Path(file_okay=True, dir_okay=False, writable=True, ),
                nargs=1)
@click.option('--estimator', metavar='ESTIMATOR',
              type=click.Choice(['logistic', 'svm', 'tree', 'random', 'deep']),
              help='The estimator to use')
@click.option('--features', metavar='FEATURES',
              type=click.Choice(['numeric', 'text', 'both']),
              help='The types of features to use')
@click.option('--block-text/--no-block-text', default=False,
              help='Whether to only use block text as a textual feature(default false)')
@click.option('--blocks/--no-blocks', default=True,
              help='Whether to only use blocks for training(default true)')
@click.option('--external-folds', metavar='N_FOLDS TOTAL_FOLDS',
              type=click.Tuple([int, int]), default=(10, 10),
              help='The number of folds to use and the total folds for the external loop(default 10 10)')
@click.option('--internal-folds', metavar='N_FOLDS TOTAL_FOLDS',
              type=click.Tuple([int, int]), default=(10, 10),
              help='The number of folds to use and the total folds for the internal loop(default 10 10)')
@click.option('--n-iter', metavar='N_ITER', type=click.INT,
              default=20, help='The number of iterations for the internal randomized search(default 20)')
@click.option('--n-jobs', metavar='N_JOBS', type=click.INT,
              default=-1, help='The number of jobs to start in parallel(default -1)')
@click.option('--random-seed', metavar='RANDOM_SEED', type=click.INT,
              default=42, help='The random seed to use')
def evaluate(dataset, output, estimator, features, blocks, external_folds, internal_folds,
             n_iter, n_jobs, random_seed, block_text):
    """Evaluate the expected f1-score with nested CV"""
    # unpacking the fold numbers
    internal_n_folds, internal_total_folds = internal_folds
    external_n_folds, external_total_folds = external_folds
    
    # seed the random number generator
    click.echo('SEEDING THE RANDOM NUMBER GENERATOR...')
    np.random.seed(random_seed)
    # TODO: find a fix for this, but for the moment,
    # there is no other solution than using tf just in the worker
    # tf.set_random_seed(random_seed)

    # load the dataset
    click.echo('LOADING THE DATASET...')
    estimator, param_distributions = get_param_grid(estimator, features)  # get the appropriate
    X, y, groups = get_ordered_dataset(dataset, blocks_only=blocks, shuffle=True, block_text=block_text)

    # training the model
    click.echo('TRAINING THE MODEL...')
    scores, cv = nested_cv(estimator, X, y, groups, param_distributions=param_distributions,
                           n_iter=n_iter, internal_n_folds=internal_n_folds,
                           internal_total_folds=internal_total_folds, external_n_folds=external_n_folds,
                           external_total_folds=external_total_folds, n_jobs=n_jobs)

    # outputting
    click.echo('SAVING RESULTS...')
   
    output_scores = output.format(suffix='scores.csv')
    output_cv = output.format(suffix='cv.csv')
    np.savetxt(output_scores, scores)
    cv.to_csv(output_cv, index=False)
        

if __name__ == '__main__':
    cli()
