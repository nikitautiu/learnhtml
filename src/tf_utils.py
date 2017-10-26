import tensorflow as tf
import dask.dataframe as dd
import numpy as np
import pandas as pd


def make_csv_decoder(input_tensor, dtypes, convert_ints=False, **kwargs):
    """Raturns a csv_decoded tensor from the input_tensor. Requires a sample
    file to determine the types. Also automatically converts booleans"""

    # infer the types
    default_values = ['' if dtype.name in ['bool', 'object'] else dtype.type() for dtype in
                      dtypes]  # convert bools and objs to string
    if convert_ints:
        # convert ints to float
        default_values = map(lambda x: 0.0 if np.issubdtype(type(x), np.integer) else x,
                             default_values)  # convert ints to float
    default_values = [[x] for x in default_values]  # must be wrapped in a list
    decoded_tensors = tf.decode_csv(input_tensor, default_values, **kwargs)

    # replace bools with their conversions
    for i, dtype in zip(range(len(decoded_tensors)), dtypes):
        if dtype.name == 'bool':
            condition = tf.equal(decoded_tensors[i], tf.constant('True'))
            decoded_tensors[i] = tf.where(condition, tf.constant(1.0), tf.constant(0.0))

    return decoded_tensors


def make_csv_col_tensors(csv_pattern=None, csv_files=None, shuffle=True, num_epochs=10, csv_decoder_kwargs={}):
    """Returns a dict of column names and their corresponding tensors.
    `shuffle` specifies whether the files and lines should be shuffled.
    `num_epochs` specifies how many time to yield every file"""
    # read all the files fitting the specification
    if csv_pattern is not None:
        filenames = tf.matching_files(csv_pattern)
        filenames_queue = tf.train.string_input_producer(filenames, shuffle=shuffle,
                                                         num_epochs=num_epochs)
    elif csv_files is not None:
        filenames_queue = tf.train.string_input_producer(csv_files, shuffle=shuffle,
                                                         num_epochs=num_epochs)

    else:
        # rais eif no file pattern specified
        raise ValueError('either csv_files or csv_pattern has to be specified')

    reader = tf.TextLineReader(skip_header_lines=1)  # intialize the reader
    key, value = reader.read(filenames_queue)

    # read the metadata
    ddf = dd.read_csv(csv_pattern if csv_pattern is not None else csv_files[0])
    decoded_tensors = make_csv_decoder(value, ddf.dtypes, **csv_decoder_kwargs)

    # return the columns as a dict
    return {k: v for k, v in zip(ddf.columns, decoded_tensors)}



def batch_tensors(tensors, batch_size=100, num_threads=4, shuffle=True):
    """Given a list or dict of tensors, return the batched equivalent"""
    if shuffle:
        return tf.train.shuffle_batch(tensors, batch_size=batch_size, num_threads=num_threads,
                                      capacity=batch_size * 20, min_after_dequeue=batch_size * 10)

    # otherwise, simply return a batching function
    return tf.train.batch(tensors, batch_size=batch_size, capacity=batch_size * 20)


def make_csv_pipeline(csv_pattern=None, csv_files=None, feature_cols=None, label_cols=None,
                      shuffle=True, num_epochs=10, batch_size=100, num_threads=4):
    """Given a glob pattern or a list of csv files, return a tensor of features
    and one of labels. Behind the scenes, the function builds a pipeline
    to extract the data with the parameters given to it.

    The numbered of threads only applies when shuffle is True, as it needs to be 1
    for the batching to be deterministic."""

    with tf.name_scope('input_pipeline') as scope:
        # make the columns
        tens_dict = make_csv_col_tensors(csv_pattern, csv_files, shuffle=shuffle,
                                         num_epochs=num_epochs, csv_decoder_kwargs={'convert_ints': True})

        # select the columns
        feature_tensors = [tens for key, tens in tens_dict.items() if key in feature_cols]
        label_tensors = [tens for key, tens in tens_dict.items() if key in label_cols]

        features, labels = tf.stack(feature_tensors), tf.stack(label_tensors)  # stack tensors

        # batch it, either shuffled or normal
        # keep at least 10 more batches in the queue
        tensor_list = [features, labels]
        return batch_tensors(tensor_list, batch_size, num_threads, shuffle)



