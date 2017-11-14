# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# skealrn
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

# tesnsorflow
import tensorflow as tf

def get_pred(estimator, input_fn, hooks=[]):
    """Given an input function and an estimator, return the d predicted labels"""
    # get prdeiction
    predicted_list = [pred['class_ids'] for pred in estimator.predict(input_fn, hooks=hooks)]
    pred_array = np.stack(predicted_list, axis=0).ravel()  # concatenate them to one

    return pred_array


def get_exp(estimator, input_fn, hooks=[]):
    """Given an input function and an estimator, return the expected labels"""
    # consume label tensors
    # not necessarily the most elegant solution but works
    label_tens = input_fn()[1]
    expected_list = []
    with tf.Session() as sess:
        for hook in hooks:
            hook.after_create_session(sess, None)  # must be ran
        # get each element of the training dataset_dragnet until the end is reached
        while True:
            try:
                elem = sess.run(label_tens)
                expected_list.append(elem)
            except tf.errors.OutOfRangeError:
                break

    expected_array = np.concatenate(expected_list).ravel()
    return expected_array


def get_metrics(estimator, input_fn, init_hooks=[], exp_arr=None):
    """Given an estimator, the input function, optionally its hooks,
    return the classification metrics for the positive class. If precomputted,
    one can pass the expected array alongside."""
    if exp_arr is None:
        exp_arr = get_exp(estimator, input_fn, init_hooks)
    pred_arr = get_pred(estimator, input_fn, init_hooks)
    class_rep = precision_recall_fscore_support(exp_arr, pred_arr)  # get the class report
    acc_score = accuracy_score(exp_arr, pred_arr)

    # build te stats and return
    return {
        'accuracy': acc_score,
        'precision': class_rep[0][1],  # only for the positive class
        'recall': class_rep[1][1],
        'f1-score': class_rep[2][1],
        'support': class_rep[3][1],
        'support_other': class_rep[3][0]
    }


def train_eval_loop(estimator, train_input_fn_and_hook, num_epochs=1000, start_epoch=0, epoch_step=1,
                    eval_input_fns_and_hooks={}):
    """Given a train input fn and hook and the same kind of pairs for the evaluation
    sets, return a dictionary of metrics for each set after every epoch."""

    # precache exp_arr
    expected_arrs = {
        eval_set_name: get_exp(estimator, eval_set_fn, hooks=[eval_set_hook])
        for eval_set_name, (eval_set_fn, eval_set_hook) in eval_input_fns_and_hooks.items()
    }

    # initialize the metrics
    metrics = {name: [] for name in eval_input_fns_and_hooks.keys()}

    # do the loop
    for epoch in range(1, num_epochs + 1):
        # train for one epoch
        estimator.train(train_input_fn_and_hook[0], hooks=[train_input_fn_and_hook[1]])

        print('\nEVALUATION\n', '=' * 40)
        # evaluate on sets
        for eval_set_name, (eval_set_fn, eval_set_hook) in eval_input_fns_and_hooks.items():
            evaluated_metrics = get_metrics(estimator, eval_set_fn, init_hooks=[eval_set_hook],
                                            exp_arr=expected_arrs[eval_set_name])
            evaluated_metrics['epoch'] = start_epoch + epoch * epoch_step

            print(eval_set_name, " ---- ", evaluated_metrics, )  # print for progress check
            metrics[eval_set_name].append(evaluated_metrics)
        print('=' * 40, '\n')

    return metrics


def plot_metric(train_stats, validation_stats, test_stats, metric_name, smoothing=21):
    """Given the scores for the train, validation and test set and a metric name
    plot the validation curves."""
    # get the best epoch for validation
    best_epoch = validation_stats.loc[validation_stats['f1-score'].idxmax(), 'epoch']
    best_index = test_stats['epoch'] == best_epoch

    # do a bit of smoothing
    smooth_train = train_stats.rolling(center=False, window=smoothing).mean().dropna()
    smooth_validation = validation_stats.rolling(center=False, window=smoothing).mean().dropna()

    # do the plotting
    fig, ax = plt.subplots(figsize=(11, 7))
    smooth_train.plot(x='epoch', y=metric_name, ax=ax,  alpha=0.8, label='train', linewidth=1)
    smooth_validation.plot(x='epoch', y=metric_name, ax=ax, alpha=0.8, label='validation', linewidth=1)
    plt.plot(best_epoch, test_stats.loc[best_index, metric_name], 'o', label='test(for best validation)', alpha=0.8)

    plt.ylim(ymax=1.01)
    plt.title(metric_name)
    plt.legend(loc='lower right')
    sns.despine(fig=fig, ax=ax)
    
    return fig, ax