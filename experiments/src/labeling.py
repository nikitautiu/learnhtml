import re

import numpy as np
import pandas as pd
from dask import dataframe as dd
from lxml import etree

from utils import get_domain_from_url


def match_urls(df, url_regex):
    """Returns a datarame of all the urls, telling whther they are matched or not"""
    url_pattrn = re.compile(url_regex)
    url_filter = lambda url: bool(url_pattrn.match(url))

    df.loc[:, 'url_match'] = df.loc[:, 'url'].apply(url_filter)

    return df


def label_pages(df, url_regex, rules):
    """Returns a DataFrame wherhe each row represents a tag
    and there are columns that specify whther the tag has a label or not.

    Rules is a dictionary of {label: xpath} which tells the xpath to match
    tags with the corresponding label.
    """
    df.loc[:, 'tree'] = df['html'].apply(etree.HTML)  # get the nodes
    df = match_urls(df, url_regex)  # match the urls

    for label, xpath in rules.items():
        # add columns with the tags matching the xpaths
        label_tag_col_name = '{0}_label_tags'.format(label)
        df.loc[:, label_tag_col_name] = None  # set it to the list of tags

        # the series of lists of tags that match the rule
        # applied only to matching ulrs
        tag_series = df.loc[df['url_match'], 'tree'].apply(lambda tree: tree.xpath(xpath))
        df.loc[df['url_match'], label_tag_col_name] = tag_series

    # iterate over rows, explode into component tags and mark them
    # as being labeled or not
    pages_labels = []  # the row values
    for row_data in df.iterrows():
        row = row_data[1]  # ignore the index, which is on the first pos

        # a series of the tags
        tags_series = pd.Series(list(row['tree'].iter()))
        label_cols = []

        for label, xpath in rules.items():
            label_tag_col_name = '{0}_label_tags'.format(label)
            label_col_name = '{0}_label'.format(label)

            if row['url_match']:
                # if the url is among the sought ones
                # chekc if the tags are in the current list of xpath-ed tags
                label_series = tags_series.apply(lambda tag: tag in row[label_tag_col_name])

            else:
                label_series = pd.Series(data=np.zeros(tags_series.size, dtype=bool))

            label_series.name = label_col_name  # rename it to the final name
            label_cols.append(label_series)

        # add the tag paths for identification
        tag_paths = tags_series.apply(lambda elem: row['tree'].getroottree().getpath(elem))
        tag_paths.name = 'path'

        # cocnatenate the labels for the current page
        page_label_df = pd.concat(label_cols + [tag_paths], axis='columns')

        # also the url
        page_label_df.loc[:, 'url'] = row['url']

        # page_label_df['url'] = row['url']  # add the url
        pages_labels.append(page_label_df)

    # return the veritcally stacked values of all the pages
    return pd.concat(pages_labels, axis='rows', ignore_index=True)


def label_data(df, rules):
    """Returns a df of all the labels given a set of url filters
    and tag xpaths"""
    results = pd.concat([label_pages(df, rule['url_regex'], rule['xpaths']) for rule in rules], axis='columns')

    # workaround because we cannot filter duplicates
    label_names = set(filter(lambda x: 'label' in x, list(results.columns)))
    grouped_labels = results.loc[:, label_names].groupby(results.loc[:, label_names].columns, axis='columns').any()  # true if any
    grouped_labels = grouped_labels.astype(int)  # better representation

    # remove all urls and paths other than one each
    paths = results.loc[:, 'path'].iloc[:, 0]
    urls = results.loc[:, 'url'].iloc[:, 0]
    return pd.concat([grouped_labels, urls, paths], axis='columns').set_index(['url', 'path'])


def get_stats(df):
    """Given a dataframe, either dask or pandas and returns
    satistics about how many labels are of each type for each site."""
    # get the number of labels for each page
    sum_df = df.groupby('url').sum().reset_index()
    # get the domains
    sum_df = sum_df.assign(domain=sum_df['url'].apply(lambda x: get_domain_from_url(x), meta=('domain', str)))

    # get how many extracted pages per domain
    domain_df = sum_df.groupby('domain').count()['url']

    # get how many pages have each label per domain
    dom_label_df = (sum_df > 0).groupby('domain').sum()

    # get how many labels per domain
    dom_tot_labl_df = sum_df.groupby('domain').sum()

    return domain_df.compute(),  dom_tot_labl_df.compute(), dom_label_df.compute()


def label_scraped_data(input_file, rules):
    """Given a csv of html, label the the data and """
    # stupid workaround, defeats the point
    ddf = dd.from_pandas(pd.read_csv(input_file), chunksize=25)

    # works just like that <3
    return ddf.map_partitions(lambda df: label_data(df, rules))