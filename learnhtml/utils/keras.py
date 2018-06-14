import copy
import tempfile
import threading
import types

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.sparse import issparse
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import class_weight


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def sparse_generator(X, Y, batch_size=128, shuffle=True):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)

    if Y is None:
        # ugly workaround
        Y = np.zeros(X.shape[0], dtype=bool)

    counter = 0
    while True:
        batch_index = sample_index[batch_size * counter:min(batch_size * (counter + 1), X.shape[0])]
        X_batch = np.zeros((len(batch_index),) + X.shape[1:])
        y_batch = np.zeros((len(batch_index),) + Y.shape[1:])

        for i, j in enumerate(batch_index):
            X_batch[i] = X[j].toarray()
            y_batch[i] = Y[j]

        counter += 1
        if Y is not None:
            yield X_batch, y_batch
        else:
            yield X_batch  # prediction batch

        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def create_model(hidden_layers, nb_features, activation='relu', dropout=None, optimizer='adagrad', opt_params={}):
    """Crete a keras sequential model with the given hidden layer sizes,
    activation, dropout and optimizer"""
    from keras import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout

    if type(optimizer) is not str:
        optimizer = optimizer(**opt_params)

    # create model
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=activation, input_dim=nb_features))
    for size in hidden_layers[1:]:
        model.add(Dense(size, activation=activation))
        if dropout is not None:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def constrain_memory():
    """Make the tensorflow backend use only as much memory as it needs"""
    from keras import backend as K

    if 'tensorflow' == K.backend():
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))


class Metrics(Callback):
    def __init__(self, validation_data, batch_size, *args, prefix='', **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_data = validation_data
        self._batch_size = batch_size
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict_generator(
            sparse_generator(self._validation_data[0], None, self._batch_size, shuffle=False),
            steps=np.ceil(self._validation_data[0].shape[0] / self._batch_size)
        )

        predict = np.round(np.asarray(preds))
        target = self._validation_data[1]
        results = {
            'precision': precision_score(target, predict),
            'recall': recall_score(target, predict),
            'f1': f1_score(target, predict)
        }
        print(' - '.join('{}{}: {}'.format(self.prefix, name, val) for name, val in results.items()))

        for name, val in results.items():
            logs['{}{}'.format(self.prefix, name)] = val


class KerasSparseClassifier(KerasClassifier):
    """KerasClassifier workaround to support sparse matrices"""

    def fit(self, x, y, **kwargs):
        """ adds sparse matrix handling """
        from keras.utils import to_categorical
        from keras import Sequential

        if not issparse(x):
            return super().fit(x, y, **kwargs)

        ############ adapted from KerasClassifier.fit ######################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)
        ### fit = fit_generator
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)

        ############################################################
        validation_data_ = fit_args.pop('validation_data', None)
        fit_args.pop('shuffle', None)

        if validation_data_ is not None:
            hist = self.model.fit_generator(
                sparse_generator(x, y, self.sk_params["batch_size"], shuffle=self.sk_params['shuffle']),
                steps_per_epoch=np.ceil(x.shape[0] / self.sk_params["batch_size"]),
                validation_data=sparse_generator(*validation_data_, shuffle=False),
                validation_steps=np.ceil(validation_data_[0].shape[0] / self.sk_params["batch_size"]),
                **fit_args)
            return hist

        hist = self.model.fit_generator(
            sparse_generator(x, y, self.sk_params["batch_size"], shuffle=self.sk_params['shuffle']),
            steps_per_epoch=np.ceil(x.shape[0] / self.sk_params["batch_size"]),
            **fit_args)
        return hist

    def predict_proba(self, x):
        """ adds sparse matrix handling """
        if not issparse(x):
            return super().predict_proba(x)

        preds = self.model.predict_generator(
            sparse_generator(x, None, self.sk_params["batch_size"], shuffle=False),
            steps=np.ceil(x.shape[0] / self.sk_params["batch_size"])
        )
        return preds

    def predict(self, x, **kwargs):
        if not issparse(x):
            return super().predict(x)

        proba = self.predict_proba(x, **kwargs)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')


class MyKerasClassifier(KerasSparseClassifier):
    """Custom KerasClassifier
    Ensures that we can use early stopping and checkpointing
    """

    def __init__(self, *args, patience=10, expiration=-1, checkpoint_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sk_params['patience'] = patience
        self.sk_params['checkpoint_file'] = checkpoint_file
        self.sk_params['expiration'] = expiration  # number of predicts after which to delete the model

    def fit(self, X, y, **kwargs):
        # local imports, needed for multiprocessing
        import keras
        from keras import backend as K
        constrain_memory()

        # delete model flag. tells the estimator to delete the model after a said number of turns
        # after finishing the first predict after a fit. prevents memory leaks?
        self._predict_turns = 0

        # cleanup the memory. We can't run models in a parallel anyway, so at least, prevent
        # the huge memory leak
        if 'tensorflow' == K.backend():
            K.clear_session()

        # declare the additional kwargs to pass done to the classifier
        additional_sk_params = {}

        # leave a 10% chunk out on which to do validation
        additional_sk_params['validation_data'] = self.sk_params.get('validation_data', None)
        if additional_sk_params['validation_data'] is None:
            val_point = int(X.shape[0] * .9)
            additional_sk_params['validation_data'] = (X[val_point:, :], y[val_point:])
            X = X[:val_point, :]
            y = y[:val_point]

        # try to get the checkpoint file, otherwise use a temporary
        checkpoint_file = self.sk_params.get('checkpoint_file', None)
        is_tmp = False
        if checkpoint_file is None:
            # create a temprorary file to save the checkpoint to
            is_tmp = True
            tmp_file = tempfile.NamedTemporaryFile()
            checkpoint_file = tmp_file.name

        metrics = Metrics(additional_sk_params['validation_data'], 1024, prefix='val_')
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_f1', min_delta=0.0001,
                                                      patience=self.sk_params.get('patience', 10),
                                                      verbose=1, mode='max')
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_f1', verbose=1, save_best_only=True,
                                                     mode='max')

        # set the calbacks per fit method, this ensures that each clone has its own callbacks
        self.sk_params['nb_features'] = X.shape[1]
        self.sk_params['batch_size'] = 1024
        self.sk_params['callbacks'] = [metrics, checkpoint, early_stopper]

        if self.sk_params.get('class_weight', None) == 'balanced':
            weights = class_weight.compute_class_weight('balanced', [0, 1], y)
            additional_sk_params['class_weight'] = dict(enumerate(weights))

        # update the params wih the injected ones
        kwargs.update(additional_sk_params)
        super().fit(X, y, **kwargs)

        # reaload from checkpoint
        self.model.load_weights(checkpoint_file)

        if is_tmp:
            # if it is temporary, delete it at the end
            tmp_file.close()

    def predict(self, *args, **kwargs):
        # TODO: this is just the most horrid workaround ever
        # wrapper that deletes the model if the estimator reaches EXPIRATION
        # the rationale is that after finishing fitting, girdsearch does 2
        # predictions, after which the model remains in memory, causing a memory leak

        result = super().predict(*args, **kwargs)
        self._predict_turns += 1

        if self._predict_turns == self.sk_params['expiration']:
            from keras import backend as K
            if 'tensorflow' == K.backend():
                import tensorflow as tf
                K.clear_session()  # clear the session just for good measure
                tf.reset_default_graph()  # this is needed for the python state
            del self.model

        return result
