import os
import re

import pandas as pd
from dask import dataframe as dd
from lxml import etree

from learnhtml.dataset_conversion import lcs
from learnhtml.dataset_conversion.blocks import Blockifier, simple_tokenizer

NON_CONTENT_BLOCK_RATIO = 1e-10


def read_dir_file(file, directory):
    """Read a file from a given directory"""
    with open(os.path.join(directory, file)) as fin:
        return fin.read()  # return the entire content


def convert_dragnet_dataset(directory, prefix='', npartitions=16):
    """Returns a csv dataset_dragnet from a dragnet one(or cleaneval).
    The urls are encoded as file://{suffix}{filename}"""
    html_dir = os.path.join(directory, "HTML")
    html_files = [file for file in os.listdir(html_dir) if file.endswith(".html")]

    ddf = dd.from_pandas(data=pd.DataFrame({'file': html_files}), npartitions=npartitions)
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


def byte_tokens(text):
    """Returns the tokens for the text, converted to bytes"""
    return [tok.encode('utf-8') for tok in simple_tokenizer(text)]


def get_block_ratios(html, gold_standard):
    """Given some html, return the" dataframe of
    paths and ratios of inclusion in the gold standard.
    """
    # https://github.com/seomoz/dragnet/blob/master/dragnet/data_processing.py
    # check that link out and see how they do it wiht a threshold
    # We get all the tokens from the extracted tokens and compute
    # longest common substring with all the tokens from the gold standard
    # this way, we preserve order as well. then we return for each block
    # the percentage of its tokens that were in this substring

    # we will transform the str to bytes, otherwise, lxml complains
    # for some edge case when the encoding is specified in the document
    root = etree.HTML(html.encode('utf-8'))  # get the nodes
    extracted_blocks = Blockifier.blocks_from_tree(root, do_css=False)  # get blocks
    tree = extracted_blocks[0].features['block_start_element'].getroottree() if len(extracted_blocks) != 0 else None

    # get all the extracted block tokens and their corresponding ids(path, tokens)
    block_tokens = [(tree.getpath(blk.features['block_start_element']), byte_tokens(blk.text))
                    for blk in extracted_blocks]

    # get the extracted and gold standard tokens concatendated
    all_gold_tokens = sum([byte_tokens(gold_block) for gold_block in gold_standard], [])
    all_block_tokens = sum((blk_toks for _, blk_toks in block_tokens), [])  # concatenate all

    # check which of the extracted blocks elong to the gold standard(common substring)
    token_inclusions = lcs.check_inclusion(all_block_tokens, all_gold_tokens)
    block_token_ids = sum(([ind] * len(blk_toks) for
                           ind, (_, blk_toks) in enumerate(block_tokens)), [])

    # compute the number of tokens for each block that are in the gold standad
    num_tokens_in_gold = [0 for _ in extracted_blocks]
    for tok_block_id, tok_incl in zip(block_token_ids, token_inclusions):
        num_tokens_in_gold[tok_block_id] += tok_incl

    # get the ratios
    block_incl_ratio = [num_toks / len(blk_toks) if len(blk_toks) != 0 else 0.
                        for num_toks, (_, blk_toks) in zip(num_tokens_in_gold, block_tokens)]

    # return (path, ratio)
    return [(path, block_incl_ratio)
            for (path, _), block_incl_ratio in zip(block_tokens, block_incl_ratio)]


def get_text_block(html):
    """Given some html, return the" dataframe of
    paths and their corresponding block texts
    """
    root = etree.HTML(html.encode('utf-8'))  # get the nodes
    extracted_blocks = Blockifier.blocks_from_tree(root, do_css=False)  # get blocks
    tree = extracted_blocks[0].features['block_start_element'].getroottree() if len(extracted_blocks) != 0 else None

    # get all the extracted block tokens and their corresponding ids(path, tokens)
    block_tokens = [tree.getpath(blk.features['block_start_element']) for blk in extracted_blocks]
    block_texts = [blk.text for blk in extracted_blocks]

    # return (path, text)
    return [(path, block_text) for path, block_text in zip(block_tokens, block_texts)]


def get_ratios_per_html(html, gold_standard):
    """Given an html text and the gold standard blocks,
    return a dataframe of (path, percentage)"""
    ratio_dict = {path: ratio for path, ratio in get_block_ratios(html, gold_standard=gold_standard)}
    text_dict = {path: text for path, text in get_text_block(html)}

    # get the html tree
    root = etree.HTML(html.encode('utf-8'))  # get the nodes
    tree = root.getroottree()  # so we can extract paths
    all_paths = (tree.getpath(node) for node in root.iter())

    paths, ratios, block_text = zip(*((path,
                                       ratio_dict.get(path, NON_CONTENT_BLOCK_RATIO),
                                       text_dict.get(path, ''))
                                      for path in all_paths))
    return pd.DataFrame(data={'path': paths, 'ratio': ratios, 'block_text': block_text})


def extract_ratios_from_df(df):
    """Given a dataframe of htmls, urls, and blocks return
    a dataframe of nodes and their match percentages with corresponding blocks

    :returns a lazy dataframe with the columns in this order(path, ratio, block_text, url)
    """
    grouped = df.groupby(level=0)[['html', 'url', 'gold_standard']]  # group by unique default index

    # apply receives each group as a Series if we are applying to a series
    # or as a Dataframe in this case(with a single row - that's why we use `iat[0]`)
    result = grouped.apply(lambda x: get_ratios_per_html(x['html'].iat[0],
                                                         x['gold_standard'].iat[0]).assign(url=x['url'].iat[0]))
    return result[['path', 'ratio', 'block_text', 'url']].reset_index(drop=True)  # drop the multiindex


def extract_ratios_from_ddf(ddf):
    """The same as the df version, but works with
    dask dataframes instead."""
    # we basicaly abuse map_partition's ability to expand indexes for lack of a working
    # groupby(level) in dask
    return ddf.map_partitions(extract_ratios_from_df, meta={'path': str, 'ratio': str, 'url': str}).clear_divisions()


def convert_dataset(directory, prefix, cleaneval=False, ratio_threshold=0.1, label_name='content',
                    return_ratios=False, return_extracted_blocks=False):
    """Given a directory for a dragnet-style dataset, return
    the `url,html` and the label dataframe. Can specify the
    threshold for the ratios above which to consider a tag content
    and also whether the dataset is or not CleanEval.

    An additional label with the name 'is_extracted_block' can be returned specifying
    whether the tag is corresponding to the block in Peters definition
    """
    html_ddf = convert_dragnet_dataset(directory, prefix)  # get the htl content
    html_ddf['gold_standard'] = html_ddf['file'].apply(get_blocks_for_file, directory=directory,
                                                       cleaneval=cleaneval, meta=('gold_standard', object))

    # assign the label to the tags that have their ratios greater than the threshold
    path_ratio_ddf = html_ddf.map_partitions(extract_ratios_from_df,
                                             meta=[('path', str), ('ratio', float),
                                                   ('block_text', str), ('url', str)]).clear_divisions()

    # get the blocks and overall extracted blocks
    path_ratio_ddf[label_name + '_label'] = (path_ratio_ddf['ratio'] > ratio_threshold)
    path_ratio_ddf['is_extracted_block'] = (path_ratio_ddf['ratio'] != NON_CONTENT_BLOCK_RATIO)

    # return the content and the labels
    return html_ddf[['url', 'html']], \
           path_ratio_ddf[['url', 'path', label_name + '_label'] + (['ratio'] if return_ratios else []) + (
               ['is_extracted_block', 'block_text'] if return_extracted_blocks else [])]
