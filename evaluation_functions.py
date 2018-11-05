import sys
import pickle
from collections import defaultdict
from time import perf_counter

import numpy as np
from tqdm import trange

from scipy.stats import norm

# TODO: Since these break the "y_pred is a column vector convention", is there
# any point in using them as sklearn metrics?
# Maybe it's better to break away from sklearn API
def mean_interval_size(y_true, y_interval):
    """
    Calculates the mean interval size from a interval prediction array.
    :param y_true: Not used, here for compatibility with scorer
    :param y_interval: A n x 2 Numpy array. First column is lower interval, second is upper
    :return: The average size of the intervals
    """
    # y_true is not used, but we keep it to maintain function signature as scorers expect it

    interval_size = y_interval[:, 1] - y_interval[:, 0]  # Guaranteed to be > 0

    return np.average(interval_size)


def mean_error_rate(y_true, y_interval):
    """
    Calculates the mean error rate in the provided intervals
    :param y_true: A numpy column array of true values
    :param y_interval: A n x 2 Numpy array. First column is lower interval, second is upper
    :return: The ratio of values in y_true that are within their corresponding interval in y_interval
    """

    wrong_intervals = ((y_true < y_interval[:, 0]) | (y_true > y_interval[:, 1])).sum()

    return wrong_intervals / y_true.shape[0]


def prequential_interval_evaluation(estimator, X, y, confidence, scoring, window_size=1000, verbose=0, additional_output=None):
    """
    Prequential evaluation of an estimator by testing then training with
    each example in sequence. If a window size is set the average of a tumbling window
    is reported.

    :param estimator: Has to support a partial_fit function
    :param X: numpy array
        Feature data
    :param y: numpy column vector
        Labels
    :param confidence: float
        The desired confidence level for the predictions, (0, 1.0).
    :param scoring: callable or dict
        If a callable, it should take y_true and y_predicted_interval (Numpy arrays) as arguments
        and return a scalar metric.
        dicts should have a mapping metric_name : callable (as above). can be used to report multiple
        metrics.
    :param window_size: int
        The size of the tumbling window, we average the metric(s) every x data points
    :param verbose: int
        If > 0 will display experiment progress bar. If > 1 will also print statistics per window.
    :param additional_output: None or Pathlib object.
        When provided, will write additional information about the run to files. These include the prediction and
        interval estimates, timings, and serialized model size.
    :return: List or dict
        If scoring is a callable, will return a list of scores, size should be ceil(n_samples / window_size)
        If scoring is a dict, will return a dict {metric_name: list of scores}, list size should be
        ceil(n_samples / window_size)
    """
    n_samples = X.shape[0]

    if isinstance(scoring, dict):
        # Scores are dicts {metric_name: list of scalar scores}
        window_scores = defaultdict(list)
        test_scores = defaultdict(list)
    else:
        # Scores are lists of scalar scores
        window_scores = []
        test_scores = []
    window_elements = 0
    window_count = 0
    total_windows = int(np.ceil(n_samples / window_size))
    window_start = perf_counter()

    def predict_interval(estimator_, data, confidence_):
        """
        Returns a an interval prediction from an estimator that provides the std of a normal distribution
        around the prediction.
        :param estimator: An estimator that provides a .predict(X, return_std=True) function, and returns
        the mean and std of a normal distribution.
        :param data: A numpy array of features
        :param confidence_: The desired confidence level, i.e. 1 - desired_error.
        :return: A prediction interval as a numpy array of floats.
        """
        assert np.isscalar(confidence_), "Confidence should be a scalar"
        if hasattr(estimator_, 'vw_predict_interval'):  # Duck typing for the function, defined for VW but not for MF
            interval_tuple = estimator.vw_predict_interval(data)
        else:
            ensemble_mean, std = estimator_.predict(data, return_std=True)
            std += 1e-6  # Avoid NaNs. TODO: Better solution? How are std=0 produced??
            interval_tuple = norm.interval(confidence_, loc=ensemble_mean, scale=std)

        return np.concatenate((interval_tuple[0][:, np.newaxis], interval_tuple[1][:, np.newaxis]), axis=1)

    total_duration = 0
    pred_file = None
    if additional_output is not None:
        pred_file = additional_output.open('w')
    for i in trange(n_samples, disable=(not verbose)):
        if i == 0:
            # sklearn does not allow prediction on an untrained model
            estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])
        y_interval = predict_interval(estimator, X[i, np.newaxis], confidence)
        y_true = y[i, np.newaxis]
        if pred_file is not None:
            pred_file.write("{}, {}\n".format(y_interval.flatten(), y_true))
        if isinstance(scoring, dict):
            for score_name, score_func in scoring.items():
                window_scores[score_name].append(score_func(y_true, y_interval))
        else:
            window_scores.append(scoring(y_true, y_interval))
        if i == 0:
            continue
        window_elements += 1  # Easier than checking inside window_scores
        # We add a final result every time we have gather window_size values,
        # or we've reached the end of the data (regardless of number of points in window)
        if window_elements == window_size or i == n_samples - 1:
            window_count += 1
            if isinstance(scoring, dict):
                assert isinstance(window_scores, dict)
                for score_name, score_list in window_scores.items():
                    window_sum = np.sum(score_list)
                    test_scores[score_name].append(window_sum / len(score_list))
                    if verbose > 1:
                        print("Window {}/{}: {}: {}".format(
                              window_count, total_windows, score_name, window_sum / len(score_list)))
            else:
                window_sum = np.sum(window_scores)
                # Divide by current window size here, window is possibly incomplete
                test_scores.append(window_sum / len(window_scores))
                if verbose > 1:
                    print("Window {}/{}: {}".format(
                        window_count, total_windows, window_sum / len(test_scores)))
            window_scores.clear()
            window_elements = 0
            window_end = perf_counter()
            window_duration = window_end - window_start
            total_duration += window_duration
            if additional_output is not None:
                test_scores["window_time"].append(window_duration)
                test_scores["evaluation time (cpu seconds)"].append(total_duration)
                model_size = sys.getsizeof(pickle.dumps(estimator))
                test_scores["model serialized size (bytes)"].append(model_size)
                if verbose > 1:
                    print("Time to process window: {} sec".format(window_duration))
                    print("Model size: {} bytes, {} MiB".format(model_size, model_size/1024**2))
            window_start = perf_counter()

        estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])

    return test_scores
