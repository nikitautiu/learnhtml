#!/usr/bin/env python3
"""This module provides a utility to easily download and label website tags."""

import click
import click_log
import dask
from dask import dataframe as dd

from learnhtml.dataset_conversion.conversion import convert_dataset
from learnhtml.log import logger


@click.group()
def cli():
    """Dataset conversion utility"""
    pass


@cli.command(short_help='merge csv files')
@click_log.simple_verbosity_option(logger)
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
        logger.info('Using {} as cache'.format(cache))
        dask.set_options(temporary_directory=cache)

    on_columns = on.split(',')  # get the columns to merge on
    result_ddf = dd.read_csv(input_files[0])  # the first one
    for in_files in input_files[1:]:
        # merge with the others
        logger.info('Merging {}'.format(in_files))
        in_file_ddf = dd.read_csv(in_files)
        result_ddf = result_ddf.merge(in_file_ddf, on=on_columns)

    # output it
    logger.info('Outputting')
    result_ddf.to_csv(output_files, index=False)

    logger.info('Done')


@cli.command(short_help='convert datasets')
@click_log.simple_verbosity_option(logger)
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
        logger.info('Outputting raw')
        html_ddf.compute().to_csv(output_directory + '/raw.csv', index=False)
    if labels:
        # output the html
        logger.info('Outputting labels')
        label_ddf.compute().to_csv(output_directory + '/labels.csv', index=False)

    logger.info('Done!')


if __name__ == '__main__':
    cli()
