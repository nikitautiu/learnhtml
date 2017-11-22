import dask.dataframe as dd
import numpy as np
import tensorflow as tf


def make_csv_decoder(input_tensor, dtypes, convert_ints=False, **kwargs):
    """Raturns a csv_decoded tensor from the input_tensor. Requires a sample
    file to determine the types. Also automatically converts booleans.
    If `convert_ints` is set to True, all values are decoded as floats(useful
    for stacking them afterwards)."""

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


class CsvDecoder(object):
    """Class for decoding csvs"""

    def __init__(self, csv_pattern):
        """Given a csv pattern, initializate the decoder
        with the relevant dtypes."""
        # do all of the ddf preprocessing here so it's not
        # repeated every decode

        ddf = dd.read_csv(csv_pattern)
        dtypes = ddf.dtypes  # get the dtypes from the csv
        self.col_names = ddf.columns  # get the col names

        self.default_values = ['' if dtype.name in ['bool', 'object'] else dtype.type() for dtype in
                               dtypes]  # convert bools and objs to string
        self.default_values = [[x] for x in self.default_values]  # must be wrapped in a list

        self.bool_cols = []
        # replace bools with their conversions
        for col_name, dtype in zip(ddf.columns, dtypes):
            if dtype.name == 'bool':
                self.bool_cols.append(col_name)

    def decode_line(self, input_tensor):
        decoded_tensors = tf.decode_csv(input_tensor, self.default_values)  # decode the csv
        tens_dict = {k: v for k, v in zip(self.col_names, decoded_tensors)}

        # decode the bools
        for col_name in self.bool_cols:
            condition = tf.equal(tens_dict[col_name], tf.constant('True'))
            tens_dict[col_name] = tf.where(condition, tf.constant(1.0), tf.constant(0.0))

        return tens_dict


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

    # read a couple of rows just to infer metadata
    df = dd.read_csv(csv_pattern if csv_pattern is not None else csv_files[0])
    decoded_tensors = make_csv_decoder(value, df.dtypes, **csv_decoder_kwargs)

    # return the columns as a dict
    return {k: v for k, v in zip(df.columns, decoded_tensors)}


def batch_tensors(tensors, batch_size=100, num_threads=4, shuffle=True):
    """Given a list or dict of tensors, return the batched equivalent"""
    if shuffle:
        return tf.train.shuffle_batch(tensors, batch_size=batch_size, num_threads=num_threads,
                                      capacity=50000, min_after_dequeue=10000)

    # otherwise, simply return a batching function
    return tf.train.batch(tensors, batch_size=batch_size, capacity=batch_size * 20)


def make_csv_pipeline(csv_pattern=None, csv_files=None, feature_cols=None, label_cols=None,
                      shuffle=True, num_epochs=10, batch_size=100, num_threads=4, in_memory=False):
    """Given a glob pattern or a list of csv files, return a tensor of features
    and one of labels. Behind the scenes, the function builds a pipeline
    to extract the data with the parameters given to it.

    The numbered of threads only applies when shuffle is True, as it needs to be 1
    for the batching to be deterministic."""

    with tf.name_scope('input_pipeline') as scope:

        # make the columns - this should be on the cpu
        # otherwise it really tanks performance
        if not in_memory:
            # csv pipeline - classic
            tens_dict = make_csv_col_tensors(csv_pattern, csv_files, shuffle=shuffle,
                                             num_epochs=num_epochs, csv_decoder_kwargs={'convert_ints': True})
            label_tensors = [tens for key, tens in tens_dict.items() if key in label_cols]
        else:
            # load them in a pandas dataframe
            df = dd.read_csv(csv_pattern or csv_files[0])[feature_cols + label_cols].astype(float).compute()
            label_df = df[label_cols]
            feature_df = df.drop(label_cols, inplace=True)

            # create the input function
            in_fn = tf.estimator.inputs.pandas_input_fn(feature_df, label_df, batch_size=batch_size,
                                                        num_epochs=num_epochs, shuffle=shuffle, num_threads=num_threads,
                                                        queue_capacity=10 * batch_size)
            tens_dict, label_tensors = in_fn()

        # select the columns
        feature_tensors = [tens for key, tens in tens_dict.items() if key in feature_cols]
        features, labels = tf.stack(feature_tensors), tf.stack(label_tensors)  # stack tensors

        if not in_memory:
            # batch it, either shuffled or normal
            # keep at least 10 more batches in the queue
            tensor_list = [features, labels]
            return batch_tensors(tensor_list, batch_size, num_threads, shuffle)

        # for in memory, it's already batched
        return features, labels


def csv_dataset(csv_pattern, label_name, num_parallel_calls=4):
    """Creates a `Dataset` object from csv patterns.
    The label name is given throught the second argument."""
    decoder = CsvDecoder(csv_pattern=csv_pattern)  # intialize the decoder, once for performances

    def decode_line(line):
        # decode them as tensors
        tens_dict = decoder.decode_line(line)

        # pop the label and return the tuple
        label = tens_dict.pop(label_name)
        return tens_dict, label

    paths = tf.data.Dataset.list_files(csv_pattern)  # use the pattern
    dataset = paths.flat_map(
        lambda filename: (tf.data.TextLineDataset(filename).skip(1)))  # skip the first line of every file
    dataset = dataset.map(decode_line, num_parallel_calls=num_parallel_calls)

    return dataset


# tf conversion stuff
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode() if type(value) is str else value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def example_constructor_from_tf_types(feature_types):
    """Given a dict of {'feature_name': tf.type} generate an example-from-dict
    function that takes a dict, like the one returned
    from `sess.run()` and returns an `Example` object"""
    # map tf.types to functions
    func_mapping = {
        tf.int64: _int64_feature,
        tf.string: _bytes_feature,
        tf.float32: _float_feature
    }
    feature_func_mapping = {
        k: func_mapping[v] for k, v in feature_types.items()
    }

    def convert(feature_values):
        return tf.train.Example(
            features=tf.train.Features(feature={k: feature_func_mapping[k](v) for k, v in feature_values.items()}))

    return convert


def feature_dict_from_tf_types(feature_types):
    """Returns a feature specification to pass
    to parse_single_example when decoding the data"""
    return {k: tf.FixedLenFeature([], v) for k, v in feature_types.items()}


def csv_to_tf_types(csv_pattern, **kws):
    """Given a csv_pattern, analyze the headers
    of those csv files and output a dict of {'feature_name': tf_type}"""
    ddf = dd.read_csv(csv_pattern, **kws)

    feature_type = {}
    for name, feat_type in zip(ddf.columns, ddf.dtypes):
        if np.issubdtype(feat_type, np.integer) or feat_type.name == 'bool':
            feat_type = tf.int64
        elif np.issubdtype(feat_type, np.float):
            feat_type = tf.float32
        else:
            feat_type = tf.string
        feature_type[name] = feat_type  # assign the feature type

    return feature_type


def convert_csv_to_tfrecords(csv_pattern, tfrecords_file, progress_bar=True):
    """Convert the files matching the given csv_pattern
    to tfrecords and output to the tfrecords files."""
    decoder = CsvDecoder(csv_pattern=csv_pattern)  # intialize the decoder, once for performances

    paths = tf.data.Dataset.list_files(csv_pattern)  # use the pattern
    dataset = paths.flat_map(lambda filename: (tf.data.TextLineDataset(filename).skip(1))).cache()
    dataset = dataset.map(decoder.decode_line, num_parallel_calls=4)

    # that dataset_dragnet no iterates over all the content of the csv returned as dictionaries
    # get the tf types
    tf_types = csv_to_tf_types(csv_pattern)
    example_constructor_func = example_constructor_from_tf_types(tf_types)  # the example creator

    # intialize the writer
    writer = tf.python_io.TFRecordWriter(tfrecords_file)

    # consume the iterator

    with tf.Session() as sess:
        next_element = dataset.make_one_shot_iterator().get_next()
        while True:
            try:
                element = sess.run(next_element)
                writer.write(example_constructor_func(element).SerializeToString())
                print('#', end='')
            except tf.errors.OutOfRangeError:
                break

    # close the writer
    writer.close()


def tfrecord_dataset(tfrecords_files, label_name, tf_types, num_parallel_calls=4):
    """Given a tfrecords file and a dictionary of
    keys to tensorflow type mapings, return a dataset_dragnet
    that reads from that outputs the a tuple of the feature
    tensor dict and the label tensor."""
    features = feature_dict_from_tf_types(tf_types)

    def decode_record(record):
        """Decodes a single example into a feature dict and label"""
        parsed_features = tf.parse_single_example(record, features)
        label = parsed_features.pop(label_name)
        return parsed_features, label

    dataset = tf.data.TFRecordDataset(tfrecords_files)
    dataset = dataset.map(decode_record, num_parallel_calls=num_parallel_calls)

    return dataset


def build_dataset(csv_pattern, add_weights=True, concat_features=True, normalize_data=False, kept_columns=None,
                  num_parallel_calls=16):
    """Given  a pattern of csv files, return the Tensorflow `Dataset`
    :param csv_pattern: the pattern of the csv files to use for the dataset
    :param add_weights: add a tensor called "weights" to the dictionary
    :param concat_features: whether to concatenate the feature tensors into
    one big tensor named "X"
    :param normalize_data: whether to do standard- test normalization on the features
    :param kept_columns: the columns of the data to keep. Can be either an iterable of the columns
        or a lambda to use as a filter
    :param num_parallel_calls: how any threads to run the pipeline operations on
    """
    ddf = dd.read_csv(csv_pattern)

    def add_weights_from_labels(tens_dict, label_tens):
        # add the weight column based on proportions
        label_proportion = ddf['content_label'].mean().compute()

        # define the constants
        positive_label_val = tf.constant(1.0, dtype=tf.float32)
        positive_proportion = tf.constant(0.5 / label_proportion, shape=(), dtype=tf.float32)
        negative_proportion = tf.constant(0.5 / (1 - label_proportion), shape=(), dtype=tf.float32)

        # the weights are added as a conditional based on the corresponding label
        weight_tens = tf.where(tf.equal(label_tens, positive_label_val),
                               positive_proportion,
                               negative_proportion)
        tens_dict['weights'] = weight_tens
        return tens_dict, label_tens

    def drop_strings(tens_dict, label):
        # drops the string columns
        return {k: v for k, v in tens_dict.items() if k not in ['url', 'path']}, label

    def normalize_features(tens_dict, label):
        # normalize values for faster convergence
        # must drop the nonnumeric cols as dask does not support numeric_only
        means = ddf.drop(['url', 'path', 'content_label'], axis=1).mean().compute()
        scale = ddf.drop(['url', 'path', 'content_label'], axis=1).std().compute()  # the scale
        return {k: (tf.to_float(v) - tf.constant(means[k], dtype=tf.float32)) / tf.constant(scale[k], dtype=tf.float32)
                for k, v in tens_dict.items()}, label

    def concat_feature_tensors(tens_dict, label):
        # concatenate everything but the weights into one big tensor
        weights = tens_dict.pop('weights')
        result = {'weights': weights, 'X': tf.stack([tf.to_float(tens) for tens in tens_dict.values()])}, label
        return result

    def drop_weights(tens_dict, label):
        # just for debug, to see fi weights are not somehow considered by evaluation
        tens_dict.pop('weights')
        return tens_dict, label

    def keep_columns(tens_dict, label):
        # filter columns based on the kept_cols attr
        filter_func = kept_columns
        if not callable(kept_columns):
            # build the filter function
            filter_func = lambda x: x in kept_columns

        # filter columns, also keep weights
        columns_to_keep = set(filter(filter_func, tens_dict.keys())) | {'weights'}
        tens_dict = {tens_name: tens_dict[tens_name] for tens_name in columns_to_keep}

        # return the new filtered dict
        return tens_dict, label

    # do the pipeline here
    dataset = csv_dataset(csv_pattern, 'content_label', num_parallel_calls=num_parallel_calls)  # decode the csv
    dataset = dataset.map(drop_strings, num_parallel_calls=num_parallel_calls)  # drop redundants
    if normalize_data:
        dataset = dataset.map(normalize_features, num_parallel_calls=num_parallel_calls)  # mean, std normalization
    dataset = dataset.map(add_weights_from_labels, num_parallel_calls=num_parallel_calls)  # add weight col

    if kept_columns is not None:
        # do filtering on the columns
        dataset = dataset.map(keep_columns, num_parallel_calls=num_parallel_calls)

    if concat_features:
        # concatenate only if specfified
        # may be omitted in case further processing is desired
        dataset = dataset.map(concat_feature_tensors, num_parallel_calls=num_parallel_calls)  # concatenate the features
        
    if not add_weights:
        # dropping the weights if neccesary
        dataset.map(drop_weights, num_parallel_calls=num_parallel_calls)

    return dataset


class IteratorInitializerHook(tf.train.SessionRunHook):
    """A hook that runs an initalization function at the beginning of the
    session to initialize an "initializable" iterator."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initiliser_func = None

    def after_create_session(self, session, coord):
        self.iterator_initiliser_func(session)


def tensor_transform_dict(dictionary):
    # transform a dictionary into a nested tensor struct
    placeholder_dict = {k: tf.placeholder(v.dtype, v.shape) for k, v in dictionary.items()}
    transformed_dict = {k: placeholder_dict[k] for k in dictionary.keys()}
    feed_dict = {placeholder_dict[k]: v for k, v in dictionary.items()}

    return transformed_dict, feed_dict


def get_tensors_and_feed_from_nested(struct):
    """Given a nested structure of in-memory np-arrays
    return an equivalent structure with the arrays replaced with
    `tf.Placeholder`s and a dictionary with the placeholders for
    keys and the corresponding arrays for values."""

    # TODO: Generalize it! for the moment expects (dict, array)
    tens_dict, feed_dict = tensor_transform_dict(struct[0])
    label_placeholder = tf.placeholder(struct[1].dtype, struct[1].shape)
    feed_dict[label_placeholder] = struct[1]

    # return the structure and feed dictionary
    return (tens_dict, label_placeholder), feed_dict


def data_from_dataset(dataset, total_size):
    """Given a dataset_dragnet and its total size return
    a all it's data a a nested structured with np.arrays
    for leafs"""

    # classic iteration, cache entire dataset_dragnet in memory
    it = dataset.batch(total_size).make_one_shot_iterator()
    values = it.get_next()

    # add them to the list. for the moment this is
    # redundant, but if the method will ever iterate until full
    # consumption, this will be needed
    elements = []
    with tf.Session() as sess:
        while True:
            try:
                elements.append(sess.run(values))
            except tf.errors.OutOfRangeError:
                break

    # return the new dataset_dragnet
    # because the graph cannot be larger than 2GB, data must be passed via placeholders
    # TODO: generalize to an arbitrary number of arrays(concatenate them)
    concatenated_struct = elements[0]

     # return the tensor slice dataset_dragnet and the feed dict
    return concatenated_struct


def get_dataset_from_tensors(struct, scope_name='data'):
    """Given a nested structure with leafs, np array, return a tensor slice
    dataset_dragnet with a feed_Dict to be used alongisde its initializable iterator"""
    with tf.name_scope(scope_name):
        placeholder_struct, feed_dict = get_tensors_and_feed_from_nested(struct)
    dataset = tf.data.Dataset.from_tensor_slices(placeholder_struct)
    # dataset_dragnet and dict
    return dataset, feed_dict


def get_input_fn_from_dataset_or_structure(dataset_or_struct):
    """Return an input function that iterates over the first argument.
    If feed_dict is given, an itializable iterator generated and
    a intialization hook returned as well alongside the function

    Whether the first argument is a dataset_dragnet or a structure is infered
    from its type.
    """
    iterator_initializer_hook = IteratorInitializerHook()
    is_dataset = issubclass(type(dataset_or_struct), tf.data.Dataset)  # check the types

    def input_fn(num_epochs=1000, batch_size=100, shuffle_buffer=1000):
        # the placholders MUST be built here, because they are tensors
        # which have to belong to the estimator's
        feed_dict = None  # for visibility
        if is_dataset:
            data = dataset_or_struct  # must be a dataset_dragnet
            data = data.cache()  # if it's not precached at least use dataset_dragnet's caching mechanism
        else:
            # it is a structure and the new dataset_dragnet must be built here
            data, feed_dict = get_dataset_from_tensors(dataset_or_struct)

        # shuffle the input if the parameter is non-zero
        if shuffle_buffer != 0:
            data = data.shuffle(buffer_size=shuffle_buffer)

            # batch, repeate, iterate
        data = data.batch(batch_size)
        data = data.repeat(num_epochs)

        # return the iterator, must be returned from here
        # so that the graph is built upon
        if is_dataset:
            iterator = data.make_one_shot_iterator()
            elems = iterator.get_next()
            return elems

        # get an initializable iterator and setup the hook
        iterator = data.make_initializable_iterator()
        elems = iterator.get_next()
        iterator_initializer_hook.iterator_initiliser_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict=feed_dict)
        return elems

    if is_dataset:
        return input_fn  # one shot iterator
    return input_fn, iterator_initializer_hook


def input_fn_from_csv(csv_pattern, precache=True, **kwargs):
    """Returns an input function and optionally an iterator
    initialization hook(if precache is True) for the files
    matching `csv_pattern`. The keyword arguments are passed
    on to the dataset_dragnet building function(`build_dataset`).

    If not precached, caching still occurs, but does not persist
    between subsequent calls of the input function."""
    uncached_dataset = build_dataset(csv_pattern, **kwargs)

    if precache:
        # cache it as a neste structure and get_input_fn_from_dataset_or_structure will
        # take care of the rest
        cached_struct = data_from_dataset(uncached_dataset, len(dd.read_csv(csv_pattern)))
        return get_input_fn_from_dataset_or_structure(cached_struct)  # this should also return the hook

    # else do not cache it and simply pass it tu get_input_fn_from_dataset
    return get_input_fn_from_dataset_or_structure(uncached_dataset)
