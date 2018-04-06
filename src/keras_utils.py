import threading
import types
import copy

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.sparse import issparse


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


class KerasSparseClassifier(KerasClassifier):
    """KerasClassifier workaround to support sparse matrices"""

    def fit(self, x, y, **kwargs):
        """ adds sparse matrix handling """
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


def create_model(hidden_layers, nb_features, activation='relu', dropout=None, optimizer='adagrad', opt_params={}):
    """Crete a keras sequential model with the given hidden layer sizes,
    activation, dropout and optimizer"""
    if type(optimizer) is not str:
        optimizer = optimizer(**opt_params)
        
    # create model
    model = Sequential()
    model.add(Dense(hidden_layers[0],  activation=activation, input_dim=nb_features))
    for size in hidden_layers[1:]:
        model.add(Dense(size,  activation=activation))
        if dropout is not None:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model