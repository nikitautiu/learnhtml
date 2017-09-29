"""This module provides a utility to easily download and label website tags."""

import json

import click
import os
import pandas as pd
import dask.dataframe as dd
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from tabulate import tabulate

from features import extract_features_from_ddf
from labeling import get_stats, label_scraped_data
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
def stats(input_file):
    """Print labelling stats"""
    ddf = dd.read_csv(input_file)
    total, total_labels, have_labels = get_stats(ddf)

    # try pretty printing
    click.secho("Total sites per domain", bold=True)
    click.echo(str(total))

    click.secho("Total labels per domain", bold=True)
    click.echo(tabulate(total_labels, headers='keys', tablefmt='psql'))

    click.secho("Pages with labels per domain", bold=True)
    click.echo(tabulate(have_labels, headers='keys', tablefmt='psql'))


@cli.command()
@click.argument('input_file', type=click.Path(file_okay=True, dir_okay=False, readable=True), metavar='INPUT_FILE')
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, readable=True), metavar='OUTPUT_DIR')
@click.option('--height', type=int, default=5, metavar='HEIGHT', help='the height of the neighbourhood')
@click.option('--depth', type=int, default=5, metavar='DEPTH', help='the depth of the neighbourhood')
def dom(input_file, output_dir, height, depth):
    """Extract the dom features and output them to a directory, in a partitioned fashion"""
    df = pd.read_csv(input_file)  # must read as pandas because dask makes a fuss about html
    oh, freqs, feats = extract_features_from_ddf(dd.from_pandas(df, chunksize=20), depth, height)

    # output all the three to csvs
    click.echo('OUTPUTING FEATURES')
    feats.to_csv(os.path.join(output_dir, 'feats-*.csv'), index=False)

    click.echo('OUTPUTING ONE-HOT')
    oh.to_csv(os.path.join(output_dir, 'oh-*.csv'), index=False)

    click.echo('OUTPUTING FREQUENCIES')
    freqs.to_csv(os.path.join(output_dir, 'freqs-*.csv'), index=False)

    click.secho('DONE!', bold=True)


if __name__ == '__main__':
    cli()
