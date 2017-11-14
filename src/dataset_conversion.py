# coding: utf-8

# standard library
import os
import re
from difflib import SequenceMatcher

# lxml
from lxml import etree

# pandas
import dask
import pandas as pd
import dask.dataframe as dd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzzprocess


def read_dir_file(file, directory):
    """Read a file from a given directory"""
    with open(os.path.join(directory, file)) as fin:
        return fin.read()  # return the entire content


def convert_dragnet_dataset(directory, prefix=''):
    """Returns a csv dataset_dragnet from a dragnet one(or cleaneval).
    The urls are encoded as file://{suffix}{filename}"""
    html_dir = os.path.join(directory, "HTML")
    html_files = [file for file in os.listdir(html_dir) if file.endswith(".html")]

    ddf = dd.from_pandas(data=pd.DataFrame({'file': html_files}), chunksize=10)
    ddf['html'] = ddf['file'].apply(read_dir_file, meta=('html', str), directory=html_dir)
    ddf['url'] = 'file://' + prefix + ddf['file']  # convert it to the proper format
    return ddf


def collapse_whitespace(strarg, remove_nl=False):
    """Returns a cleaned-up version of the block text
    It collapses  whitespaces, removes tabs, and, if specified,
    only keeps the tag delimiters(like in cleaneval) as newlines."""
    strarg = re.sub(r'\t+', ' ', strarg)  # replace tabs with spaces
    if remove_nl:
        # remove newlines, used for cleaneval
        # they will be replaced by the remove tags
        strarg = re.sub(r'\n', ' ', strarg)
    strarg = re.sub(r'<[a-zA-Z]+>', '\n', strarg)
    strarg = re.sub(r' +', ' ', strarg)  # collapse whitespace
    return strarg


def get_blocks(strarg, cleaneval=False):
    """Gets the sanitized blocks to use for fuzzy matching
    First sanitizes the entire text, then splits it, trims excessive
    whitespace and removes any null ones."""
    sanitized_str = collapse_whitespace(strarg, remove_nl=cleaneval)  # sanitize the string
    blocks = sanitized_str.split('\n')  # split each block of text
    stripped_blocks = (block.strip() for block in blocks)  # trim leading amnd trailing whitespace
    return [block for block in stripped_blocks if block]  # filter out nempty blocks


def get_blocks_for_file(filename, directory, cleaneval=False):
    """For the given filename(the html file), and the root directory of
    the dataset_dragnet, return its list of blocks. Can specify wether the dataset_dragnet is
    clean eval or not."""
    corrected_dir = os.path.join(directory, "Corrected")
    filename = filename + '.corrected.txt'
    with open(os.path.join(corrected_dir, filename)) as f:
        content = f.read()  # retrieve the content

    return get_blocks(content, cleaneval=cleaneval)  # sanitize and return


def extract_text_from_html(html):
    """Given some html, return the" dataframe of
    paths and text contents.
    """
    # we will transfor the str to bytes, otherwise, lxml complains
    # for some edge casse when the encoding is specified in the document
    root = etree.HTML(html.encode('utf-8'))  # get the nodes
    paths, texts = zip(*((node.getroottree().getpath(node),
                          '' if node.tag is etree.Comment or node.tag is etree.PI else ''.join(node.itertext()))
                         for node in root.iter()))
    return pd.DataFrame(data={'path': paths, 'text': texts})


def extract_text_from_df(df):
    """Given a dataframe of htmls and urls, return
    a dataframe of nodes and their text content.
    """
    grouped = df.groupby(level=0)[['html', 'url']]  # group by unique default index

    # apply receives each group as a Series if we are applying to a series
    # or as a Datafram in this case(with a single row)
    result = grouped.apply(lambda x: extract_text_from_html(x['html'].iat[0]).assign(url=x['url'].iat[0]))
    return result.reset_index(drop=True)  # drop the multiindex


def extract_text_from_ddf(ddf):
    """The same a s the df version, but works with
    dask dataframes instead."""
    # we basicaly abuse map_partition's ability to expand indexes for lack of a working
    # groupby(level) in dask
    return ddf.map_partitions(extract_text_from_df, meta={'path': str, 'text': str, 'url': str}).clear_divisions()


def split_text(elem):
    return [tok for tok in re.split('[\n\t ]*\n[\n\t ]*', elem) if tok != '']



BLACKLIST = {'applet', 'area', 'base', 'basefont', 'bdo', 'button', 'caption', 'fieldset',
             'fram', 'frameset', 'iframe', 'img', 'input', 'legend', 'link', 'menu', 'meta',
             'noframes', 'noscript', 'object', 'optgroup', 'option', 'param', 'script', 'select',
             'style', 'textarea', 'var', 'xmp', 'like', 'like-box', 'plusone', 'svg', 'math', 'html', 'body',
             'comment()', 'head', 'title'}


def get_node_type(path):
    """Return the node type for a given xpath"""
    slash_pos = path.rfind('/')
    end_pos = path.find('[', slash_pos)
    if end_pos != -1:
        return path[slash_pos+1:end_pos]  # return the last
    return path[slash_pos+1:]


def block_max_ratio(ser):
    """Receives a text and a series indexed by url
    in which to check for it.It returns the maximum fuzzy
    match ratio"""
    if get_node_type(ser['path']) in BLACKLIST:
        ser['ratio'] = 0.  # do not extract blacklisted frames or from head
    else:
        # get the mean of all the blocks of the current tag
        ratios = [(fuzzprocess.extractOne(text, ser['blocks']) or (0, 0))[1] / 100 for text in ser['text']]
        ser['ratio'] = sum(ratios) / len(ratios) if len(ratios) != 0 else 0.  # return the maximum
    return ser


def convert_dataset(directory, prefix, cleaneval=False, block_thresh=0.99, label_name='content',
                    return_ratios=False):
    """Given a directory for a dragnet-style dataset_dragnet, return
    the `url,html` and the label dataframe. Can specify the
    fuzzy threshold for which to consider a text corresponding to the block
    and also wether the tdataset is or not CleanEval"""
    html_ddf = convert_dragnet_dataset(directory, prefix)  # get the htl content
    html_ddf['blocks'] = html_ddf['file'].apply(get_blocks_for_file, directory=directory,
                                                cleaneval=cleaneval, meta=('blocks', object))
    text_tag_ddf = extract_text_from_ddf(html_ddf)

    # persist it for the lookup speedup
    html_df = html_ddf.compute()

    # merge the dataframes
    text_tag_ddf = text_tag_ddf.merge(html_df, on='url')[['url', 'path', 'text', 'blocks']]
    text_tag_ddf['text'] = text_tag_ddf['text'].apply(split_text, meta=('', object))
    text_tag_ddf = text_tag_ddf.apply(block_max_ratio, axis=1,
                                      meta=[('url', object), ('path', object),
                                            ('text', object), ('blocks', object),
                                            ('ratio', float)])[['url', 'path', 'ratio']]
    # assign the label to the tags that hae the grates matchin ration of over
    # value of the threshold
    text_tag_ddf[label_name + '_label'] = (text_tag_ddf['ratio'] >= block_thresh)

    # return the content and the labels
    return html_df[['url', 'html']], \
           text_tag_ddf[['url', 'path', label_name + '_label'] + (['ratio'] if return_ratios else [])]
