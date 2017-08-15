# because we don't have enough memory to actually load all
# of them in memory, we will have to iterate over their chunks
# chunk-concatenate the dataframes
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def cont_to_csv(chunk_gen, *args, **kwargs):
    """Writes chunk by chunk to a csv file.
    Given a generator, consume it and write it to a csv file.
    All the other rguments are passed to the `to_csv` method."""
    for step, chunk in enumerate(chunk_gen):
        if step == 0:
            # honor the header option for the first chunk
            chunk.to_csv(*args, header=kwargs.pop('header', True), **kwargs)
        else:
            # no header and apend mode
            chunk.to_csv(*args, mode='a', header=False, **kwargs)


def url_label_chunk_gen(label, dataset_df, oh_dfs):
    """Given a label, yield all the
    dataframe chunks of them concatenated and filtered
    to only the urls containing the label."""
    chunk_iters = zip(*(iterate_chunks(df) for df in [dataset_df] + oh_dfs))
    valid_urls = get_containing_urls(dataset_df, label)  # al the urls to use

    # labels to drop
    all_labels = set(dataset_df.filter(axis='columns', regex='^.*_label$').columns)
    dropped_labels = all_labels - set(label)

    # process them sequentially with chunks
    for chunks in chunk_iters:
        concat_chunks = filter_by_urls(pd.concat(chunks, axis='columns'), valid_urls)
        # drop the unneeded labels, rename the needed one simply to "label" and yield
        yield concat_chunks.drop(dropped_labels, axis='columns').rename({label: 'label'})


def filter_by_urls(df, urls):
    """Returns the df, filtered by """
    return df[df['url'].isin(urls)]


def get_containing_urls(df, label):
    """Returns the urls of all the pages that contain at least a
    tag with the given label."""
    grp_df = df.groupby('url')[label].any().reset_index()
    return grp_df[grp_df[label]]['url']


def iterate_chunks(df, chunksize=1000):
    """Iterates over a """
    yield from (g for _, g in df.groupby(np.arange(len(df)) // chunksize))


def onehot_df(series, prefix=''):
    """Given a series, return a one hot encoded
    dataframe of the classes, the nase are prefixed with the
    optional argument."""
    binarizer = LabelBinarizer()
    data = binarizer.fit_transform(series)
    return pd.SparseDataFrame(data=data, columns=[prefix + cls for cls in binarizer.classes_])
