import itertools
import re
from urllib.parse import urlparse

import dask.dataframe as dd
import numpy as np
import pandas as pd
from lxml import etree


def get_depth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d


def extract_depths(nodes):
    """Returns a Series of the depths of the nodes"""
    return pd.Series(data=(get_depth(node) for node in nodes))


def extract_no_children(nodes):
    """Returns a Series of the number of children for each node"""
    return pd.Series(data=(len(node.getchildren()) for node in nodes))


def extract_tag_types(nodes):
    return pd.Series(data=(node.tag if type(node.tag) is str else 'html_comment' for node in nodes))


def extract_text_len(nodes):
    """Returns the number of characters of text for the given node."""

    def get_text_len(nodes):
        for node in nodes:
            text = '' if node.tag is etree.Comment or node.tag is etree.PI else ''.join(node.itertext())
            yield len(text)

    return pd.Series(data=get_text_len(nodes))


def extract_classes(nodes):
    """Extract a Series of class lists for each node."""
    return pd.Series(data=(node.attrib.get('class', '').split() for node in nodes))


def extract_class_text(nodes):
    """Extract a Series of class lists for each node."""
    return pd.Series(data=(node.attrib.get('class', '') for node in nodes))


def extract_id_text(nodes):
    """Extract a Series of class lists for each node."""
    return pd.Series(data=(node.attrib.get('id', '') for node in nodes))


def extract_attr_len(nodes, attr_name='id'):
    """Extract a Series of bools telling whether the component
    has and id attribute or not."""
    return pd.Series(data=(len(node.attrib.get(attr_name, '')) for node in nodes))


def extract_no_classes(nodes):
    """Extracts the number of classes for each node"""
    return pd.Series(data=(len(classes) for classes in extract_classes(nodes)))


def get_sibling_pos(node):
    if node.getparent() is None:
        return 0
    return node.getparent().index(node)  # return the position


def extract_sibling_pos(nodes):
    """Returns a Series of the position of each node amongst its siblings"""
    return pd.Series(data=(get_sibling_pos(node) for node in nodes))


def get_descendants(node, depth):
    """
    Returns a dict of elements {depth: nodes}
    of what nodes are at each depth in the subtree whose root is
    the given node.

    Ex: ```
    A - B - D
     \- C - E
         \- F
          \- H
    ```
    return for A: `{1: [B, C], 2: [D, E, F, H]}`
    """
    tree = {}
    for i in range(1, depth + 1):
        # return node for the first depth
        # else the elements on the prev level
        parents = tree.get(i - 1, []) if i > 1 else [node]

        # chain the nodes on the level and insert them
        level_nodes = itertools.chain(*(parent.getchildren() for parent in parents))
        tree[i] = list(level_nodes)

    return tree


def get_ancestors(node, height):
    """Returns a list of ancestor nodes up to the given height"""
    current_node = node.getparent()
    current_height = 1
    while current_node is not None and current_height <= height:
        yield current_node
        current_node = current_node.getparent()
        current_height += 1  # increment the height


def extract_node_features(nodes):
    """Returns a dataframe of features from the nodes"""
    depth_features = extract_depths(nodes)  # depths
    sibling_pos_features = extract_sibling_pos(nodes)  # position among siblings
    tag_type_features = extract_tag_types(nodes)  # tag types
    no_classes_features = extract_no_classes(nodes)  # # of classes
    id_len_features = extract_attr_len(nodes)  # id len
    class_len_features = extract_attr_len(nodes, 'class')  # class len
    no_children_features = extract_no_children(nodes)  # # of children
    text_len_features = extract_text_len(nodes)  # text length
    classes_features = extract_classes(nodes)  # classes
    class_text_features = extract_class_text(nodes)  # class text
    id_text_features = extract_id_text(nodes)  # class text

    # series of features, and their names
    series = [depth_features, sibling_pos_features,
              tag_type_features, no_classes_features,
              id_len_features, class_len_features, no_children_features,
              text_len_features, classes_features, class_text_features, id_text_features]
    columns = ['depth', 'sibling_pos', 'tag', 'no_classes', 'id_len', 'class_len',
               'no_children', 'text_len', 'classes', 'class_text', 'id_text']
    df_items = zip(columns, series)

    return pd.DataFrame.from_items(df_items)


def iter_df_records(df):
    """Iterate over the dataframe as records"""
    yield from (rec[1] for rec in df.iterrows())


def get_empty_features():
    """Returns the null equivalent of empty features for a node"""
    return np.array([0, 0, '', 0, 0, 0, 0, 0, list(), '', ''], dtype=object)


def aggregate_features(feat_list):
    """Aggregates a list of features into one.
    Returns a list of tags, the average number of classes,
    the average number of children, the ratio of nodes with
    text and a list of all classes, and the ratio of features
    which have an id. Also, the total number of nodes."""
    if len(feat_list) != 0:
        # try to compute only i there are descendants
        no_nodes = len(feat_list)
        no_children_avg = np.array([feat['no_children'] for feat in feat_list]).mean()
        id_len_avg = np.array([feat['id_len'] for feat in feat_list]).mean()
        no_classes_avg = np.array([feat['no_classes'] for feat in feat_list]).mean()
        class_len_avg = np.array([feat['class_len'] for feat in feat_list]).mean()
        text_len_avg = np.array([feat['text_len'] for feat in feat_list]).mean()
        class_list = sum((feat['classes'] for feat in feat_list), [])
        tag_list = [feat['tag'] for feat in feat_list]

        return no_nodes, no_children_avg, id_len_avg, no_classes_avg, class_len_avg, text_len_avg, class_list, tag_list

    # return an empty set of features otherwise
    return 0, 0, 0, 0, 0, 0, list(), list()


class NodeFeatureExtractor(object):
    def __init__(self, nodes):
        # preextract them for efficiency
        self.feats_per_node = extract_node_features(nodes)
        self.feature_tree = dict(zip(nodes, iter_df_records(self.feats_per_node)))
        self.nodes = nodes

    def extract_node_features(self):
        return self.feats_per_node  # plain return

    def extract_ancestor_features(self, height):
        """Extracts features from the ancestors of the node up to the given
        height. Pads with the null if unavailable"""
        feature_names = self.feats_per_node.columns  # the names of the features
        feature_dtypes = self.feats_per_node.dtypes  # the types

        # add a feature placeholder to pad element that are high
        # enough in the tree not to have enough ancestors
        feature_rows = []
        empty_row = get_empty_features()
        for node in self.nodes:
            # traverse ancestors
            feature_list = [self.feature_tree[ancestor]
                            for ancestor in get_ancestors(node, height)]
            # pad with empty ancestor features until we get a length equal to the height
            feature_list.extend([empty_row] * (height - len(feature_list)))
            feature_rows.append(np.hstack(feature_list))

        # rename the columns
        column_names = []
        for i in range(1, height + 1):
            for name in feature_names:
                column_names.append('ancestor{}_{}'.format(i, name))

        # reconvert them to the original types
        column_dtypes = dict(zip(column_names, feature_dtypes.tolist() * height))
        return pd.DataFrame(data=np.vstack(feature_rows),
                            columns=column_names).astype(column_dtypes)

    def get_descendant_agg_feats(self, node, depth):
        """Returns the aggregate features for each level of the subtree
        as the concatenation of the features for each level"""
        per_level_tuples = []  # list of agg feats on each level
        descendants = get_descendants(node, depth)

        # iterate over levels
        for lvl in range(1, depth + 1):
            # get the features of the descendants
            desc_feat_list = [self.feature_tree[desc] for desc in descendants[lvl]]
            per_level_tuples.append(aggregate_features(desc_feat_list))

        return sum(per_level_tuples, ())  # needed for concatenation

    def extract_descendant_features(self, depth):
        """Extracts for all the nodes the descendant features"""

        feature_rows = []
        for node in self.nodes:
            # get the descendant aggregate features
            feature_rows.append(self.get_descendant_agg_feats(node, depth))

        feature_names = ['no_nodes', 'no_children_avg', 'id_len_avg',
                         'no_classes_avg', 'class_len_avg', 'text_len_avg',
                         'classes', 'tags']
        column_names = []
        for i in range(1, depth + 1):
            for name in feature_names:
                column_names.append('descendant{}_{}'.format(i, name))

        # just feed the tuples and specify the column names
        return pd.DataFrame(data=feature_rows, columns=column_names)


def extract_features_from_nodes(nodes, depth, height):
    """Helper for calling the extractor with a given height and depth"""
    extractor = NodeFeatureExtractor(nodes)
    feat_dfs = [extractor.extract_node_features()]  # a list of feature dataframes to concat

    # check whether to extract the descendant/ancestor features
    if depth > 0:
        feat_dfs.append(extractor.extract_descendant_features(depth))
    if height > 0:
        feat_dfs.append(extractor.extract_ancestor_features((height)))

    # either concatenate or not
    features = feat_dfs[0] if len(feat_dfs) == 1 else pd.concat(feat_dfs, axis='columns')
    return features


def extract_features_from_html(html, depth, height):
    """Given an html text, extract the node based features
    including the descendant and ancestor ones if depthe and
    height are respectively non-null."""
    root = etree.HTML(html.encode('utf-8'))  # get the nodes, serve bytes, unicode fails if html has meta
    features = extract_features_from_nodes(list(root.iter()), depth, height)

    # add the paths to the elements for identification
    features.loc[:, 'path'] = pd.Series((node.getroottree().getpath(node) for node in root.iter()))

    return features


def get_domain_from_url(url):
    """Returns the fully-qualified domain of an url."""
    parsed_uri = urlparse(url)
    return parsed_uri.netloc


def extract_features_from_df(df, depth, height):
    """Given a dataframe of htmls and urls, return
    a dataframe of node features, return a dataframe
    with the node features extracted also, for each node,
    add the url and path respectively"""
    feat_dfs = []
    rows = df.iterrows()
    if len(df) == 0:
        # return dummy data in non inputed(for dask)
        rows = [(0, {'html': '<html><head></head><body></body></html>', 'url': 'aaa'})]

    for index, row in rows:
        # extract the html features from each of the entries
        feat_df = extract_features_from_html(row['html'], depth, height)
        feat_df['url'] = row['url']
        feat_dfs.append(feat_df)  # add the features to the list

    # concat them all
    result_df = pd.concat(feat_dfs, ignore_index=True)
    return result_df


def count_values(lst, values):
    """Given an iterable of values and one of keys, return the count of
    the keys in the list(with 0 as default)"""
    count_dict = {val: 0 for val in values}  # for overwriting with values
    for elem in lst:
        count_dict[elem] += 1
    return count_dict


def freq_vect_series(ser):
    """Given a series whose elements are python lists, return
    a dataframe where each record is the frequency vector for a certain
    element in the list. The columns will be prefixed with the series name

    Returns a dask datagrame."""
    # reduce all to a single set of tags
    avail_tags = ser.to_bag().fold(lambda a, b: a | set(b), set.union, initial=set()).compute()
    # compute the frequencies of the given tags, pass the index as an argument to concat it to the dict
    # to preserv it
    freqcol_names = {tag_name: int for tag_name in avail_tags}
    freqs = ser.apply(lambda x: pd.Series(count_values(x, avail_tags)), meta=freqcol_names)

    # rename the columns to be prefixed with the name of the series
    col_renames = {col_name: ser.name + '_' + col_name for col_name in avail_tags}
    return freqs.rename(columns=col_renames)


def freq_vect_dataframe(ddf, freq_cols=None):
    """Given a dataframe of columns with python lists compute
    the merged dataframe of the frequency vectors returned
    by freq_vect_series."""
    # determine columns
    if freq_cols is None:
        freq_cols = ddf.columns.tolist()  # use all if unspecified

    ddfs = [freq_vect_series(ddf.loc[:, col_name]) for col_name in freq_cols]
    # basically compute all the frequency dataframes and returned the one-by-one merge result
    result = ddf.drop(freq_cols, axis=1)  # drop all but the columns to be transformed to freqs
    for current_ddf in ddfs:
        result = result.assign(**{col_name: current_ddf[col_name] for col_name in current_ddf.columns.tolist()})
    return result


def one_hot_dataframe(ddf, one_hot_cols=None):
    """Given a dask dataframe encode its columns using one-hot. Every new column will
    be prefixed with the original name.

    Returns a dask dataframe."""

    tag_cats = ddf.categorize()  # converted to categoricals
    # will be using al if None
    one_hot = dd.get_dummies(data=tag_cats, prefix=one_hot_cols, columns=one_hot_cols)
    return one_hot


def extract_features_from_ddf(ddf, depth, height):
    """Given a dask dataframe of the raw data, return the dask dataset_dragnet containing all the
    extracted features and dropping the redundant ones."""
    feat_ddf = ddf.map_partitions(lambda df: extract_features_from_df(df, depth, height),
                                  meta=extract_features_from_df(pd.DataFrame(), depth, height)).clear_divisions()
    feat_ddf = feat_ddf.categorize(['url', 'path'])

    columns = feat_ddf.columns.tolist()  # used for filtering

    # one hot encoding
    one_hot_cols = list(filter(lambda col: re.match(r'.*tag$', col), columns))
    one_hot_ddf = one_hot_dataframe(feat_ddf.loc[:, one_hot_cols + ['url', 'path']], one_hot_cols)

    # frequency vects
    freq_cols = list(filter(lambda col: re.match(r'descend.*tags$', col), columns))
    freq_ddf = freq_vect_dataframe(feat_ddf.loc[:, freq_cols + ['url', 'path']], freq_cols)

    # drop redundant cols
    classes_cols = list(filter(lambda col: re.match(r'^((descendant|ancestor)[0-9]+_)?classes$', col), columns))
    feat_ddf = feat_ddf.drop(one_hot_cols + freq_cols + classes_cols, axis='columns')
    return one_hot_ddf, freq_ddf, feat_ddf
