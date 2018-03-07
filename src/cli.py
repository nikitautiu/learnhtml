#!/usr/bin/env python3
"""This module provides a utility to easily download and label website tags."""

import json
import os

import click
import dask
import dask.dataframe as dd
import pandas as pd
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from dataset_conversion.conversion import convert_dataset
from features import extract_features_from_ddf
from labeling import label_scraped_data
from scrape.spiders.broad_spider import HtmlSpider


def run_scrape(start_urls, format, output, logfile=None, loglevel=None, use_splash=False, max_pages=100):
    """Scrapes the given urls, with the given spider settings
    and outputs to a given file
    """
    #  update the settings
    settings = get_project_settings()

    # feed export settings
    settings.set('FEED_URI', output, priority='cmdline')
    settings.set('FEED_FORMAT', format, priority='cmdline')

    # logging settings
    # click passes the arguments as null if unspecified
    if logfile:
        settings.set('LOG_ENABLED', True, priority='cmdline')
        settings.set('LOG_FILE', logfile, priority='cmdline')

    if loglevel:
        settings.set('LOG_ENABLED', True, priority='cmdline')
        settings.set('LOG_LEVEL', loglevel, prioriy='cmdline')

    crawler = CrawlerProcess(settings=settings)
    crawler.crawl(HtmlSpider, start_url=start_urls, use_splash=use_splash, follow_links=True, max_pages=max_pages)
    crawler.start()


@click.group()
def cli():
    """Dataset creation tool"""
    pass


@cli.command()
@click.option('--logfile', type=click.Path(), metavar='FILE',
              help='the file to log to')
@click.option('--loglevel', type=click.STRING, metavar='LEVEL',
              help='the log level')
@click.argument('output_file', type=click.Path(file_okay=True, dir_okay=False), metavar='OUTPUT_FILE')
@click.option('--rules', type=click.Path(file_okay=True, dir_okay=False, readable=True), metavar='RULES_FILE',
              default='rules.json', help='the json rules file')
@click.option('--start-url', type=click.STRING, default=None, help='url to start from')
@click.option('--pages', type=click.INT, metavar='PAGES', default=100, help='the number of pages to extract per domain')
def scrape(output_file, rules, logfile, loglevel, pages, start_url):
    """Scrapes and labels given an output directory and a rule file.
    The rules file expects a json file with a dictionary with the following structure:

    name: {
        urls: [],
        rules: [
            {
                url_regex: "asdas",
                xpaths: {
                    label1: "xpath1",
                    label2: "xpath2"
                }
            }
        ]
    }
    """
    if start_url:
        urls = [start_url]
    else:
        # load the json
        with open(rules) as f:
            rules_dict = json.load(f)
        urls = rules_dict['urls']

    # scrape the data
    run_scrape(urls, 'csv', output_file, use_splash=True, logfile=logfile, loglevel=loglevel,
               max_pages=pages)
    click.secho('Sucessfully scraped!', fg='green', bold=True)


@cli.command()
@click.argument('input_file', type=click.Path(file_okay=True, dir_okay=False, readable=True), metavar='INPUT_FILE')
@click.argument('output_file', metavar='OUTPUT_FILE')
@click.option('--rules', type=click.Path(file_okay=True, dir_okay=False, readable=True), metavar='[RULES_FILE]',
              default='rules.json', help='the json rules file')
def label(input_file, output_file, rules):
    """Label the tags of the html in the INPUT_FILE and
    write them in csv in OUTPUT_FILE"""
    # load the json
    with open(rules) as f:
        rules_dict = json.load(f)

    # extract labels
    labeled_df = label_scraped_data(input_file, rules_dict['rules'])
    labeled_df.to_csv(output_file, index=True)


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
@click.option('--cache', type=click.Path(dir_okay=True, file_okay=False, exists=True),
              metavar='CACHE_DIR', help='where to store cache for larger-than-memory merging',
              default=None)
@click.option('--state', type=click.INT, metavar='RANDOM_STATE',
              help='the random seed', default=42)
@click.option('--on', type=click.STRING, metavar='SPLIT_COL', help='the column to split by')
@click.argument('input_files', metavar='INPUT_FILES', nargs=1)
@click.argument('outputs', metavar='OUTPUTS', type=click.STRING, nargs=-1)
def split(cache, outputs, input_files, on, state):
    """Splits the CSV by the given column. The outputs
    are given as OUTPUT_PATTERN1 OUTPUT_PROPORTION1 OUTPUT_PATTERN2
    OUTPUT_PROPORTION2 etc."""
    # set the cache if specified

    if cache is not None:
        click.echo('Using {} as cache'.format(cache))
        dask.set_options(temporary_directory=cache)

    proportions = [int(prop) for prop in outputs[1::2]]
    proportions = [prop / sum(proportions) for prop in proportions]  # scaled
    output_csvs = outputs[::2]

    click.echo('Computing split')

    # split the columns unique values
    ddf = dd.read_csv(input_files)  # the first one
    splits = [split.compute() for split in ddf[on].unique().random_split(proportions, random_state=state)]

    # iterate over splits and output
    for split_values, out_csv in zip(splits, output_csvs):
        click.echo('Outputting to {}'.format(out_csv))
        ddf[ddf[on].isin(split_values)].to_csv(out_csv, index=False)


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


if __name__ == '__main__':
    cli()
