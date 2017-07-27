import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def pad_design_matrix(X):
    """Return the design matrix padded with 
    the constant feature"""
    aux = np.ones((X.shape[0], X.shape[1] + 1))
    aux[:, 1:] = X
    return aux


def plot_boundaries(X, Y, beta, quadratic=False):
    """Plot a 2-dimensional dataset and the class
    boundaries according to the given betas.
    If it's a two-class problem, plot the line that separates,
    otherwise, plot the hyperplane intersections."""
    for cls in set(Y):
        plt.scatter(X[Y == cls].T[0], X[Y == cls].T[1])

    # for linear boundary
    if len(set(Y)) == 2:
        # d: beta0 + X1 * beta1 + X2 * beta2 = 0
        X1 = np.linspace(0, 1, 1000)
        X2 = -(beta[1] * X1 + beta[0]) / beta[2]
        plt.plot(X1, X2)  # distinct color

    # for hyperplane bondaries
    else:
        # get pairs of betas
        for row1, row2 in itertools.combinations(list(beta.T), 2):
            X1 = np.linspace(0, 1, 1000)
            diff = row1 - row2

            # different equations for quadratics
            if quadratic:
                X2 = (diff[0] + diff[1] * X1) / (-diff[2] - diff[3] * X1)
                # add the discontinuities
                disc_X1 = -diff[2] / diff[3]

                X1 = np.append(X1, disc_X1)
                X2 = np.append(X2, np.nan)

                # insert the discontinuity and sort
                plot_vals = np.vstack([X1, X2])
                i = np.argsort(X1)
                plot_vals = plot_vals[:, i]
                X1, X2 = list(plot_vals)  # list-unwrapping
            else:
                X2 = (diff[0] + diff[1] * X1) / -diff[2]
            plt.plot(X1, X2)  # distinct color

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    sns.despine()



def generate_dataset(size, dim=2, classes=3):
    """Generate a synthetic dataset with a given
    number of classes. The classes have points in a
    gaussian distribution around uniformly distributed
    centroids"""
    # get the centroids of the classes
    centroids = np.random.uniform(size=(classes, dim), low=0.2, high=0.8)

    X_rows = []
    Y_rows = []
    for _ in range(size):
        noise = np.random.normal(size=(classes, dim), scale=0.04)
        # add gaussian noise to the centroids to get samples
        X_rows.append(centroids + noise)
        Y_rows.append(np.arange(classes))

    # stack the X and Y
    return np.vstack(X_rows), np.hstack(Y_rows)


def one_hot_encode(labels):
    """Returns the one-hot encoded matrix of
    the features"""
    label_set = set(labels)
    label_mapping = {label: num for num, label in enumerate(label_set)}
    rows = []  # the rows of the new matrix
    for label in labels:
        # compute the row to add
        if len(label_set) == 2:
            # in case there are only two albels
            # use a single column
            row = np.array([float(label == list(label_set)[0])])
        else:
            # for more than two, do proper one-hot
            pos = label_mapping[label]
            row = np.zeros(len(label_set))
            row[pos] = 1

        rows.append(row)

    return np.vstack(rows)


def ordinary_least_squares(X, Y, l=0):
    """Return the best coefficients using OLS
    Does ridge regression if lambda is specified"""
    # the design matrix has the vectors
    # as rows instead of cols
    X_pad = pad_design_matrix(X)

    # add the regularization term. unaffected if 0
    inv = np.linalg.inv(X_pad.T.dot(X_pad) + l * np.eye(X_pad.shape[1]))
    return inv.dot(X_pad.T.dot(Y))


def mse(X, beta, Y):
    """Returns the mean squared error given the data and params."""
    # mean squared error
    return np.mean((pad_design_matrix(X).dot(beta) - Y).ravel() ** 2)


def gradient_descent(theta, grad_func, alpha=0.001, epochs=100):
    """Perform gradient descent on the given values with a given
    learning rate and yield, for ever step, the optimized params."""
    yield theta
    for i in range(epochs):
        theta -= alpha * grad_func(theta)  # descend on the gradient
        yield theta


def classify(X, beta):
    """Returns the classificationsof the given data"""
    return np.argmax((beta.T.dot(X.T)), axis=0)


def classification_accuracy(Y_preds, Y_real):
    """Returns the classifiction accuracy of the predictions"""
    return np.mean(1. * (Y_preds == Y_real))


def k_fold_split(X, Y, k=5):
    """Yields all fold splits of the train data as
    (X_train, Y_train, X_test, Y_test). Defaults to
    80/20 split(aka. 5 folds)."""
    splits = np.linspace(0, X.shape[0], k + 1).astype(int)
    intervals = zip(splits[:-1], splits[1:])  # the intervals for the test data

    for start, stop in intervals:
        # the training set is the entire set excluding test
        X_train = np.vstack([X[:start], X[stop:]])
        Y_train = np.concatenate([Y[:start], Y[stop:]])
        X_test = X[start:stop]
        Y_test = Y[start:stop]
        yield (X_train, Y_train, X_test, Y_test)