import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from skgarden import MondrianForestRegressor
from scipy.io import arff
from scipy import stats
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train",
                        help="Location of the train data (arff file)")
    parser.add_argument("--test",
                        help="Location of the test data (arff file)")

    return parser.parse_args()


def load_arff_data(filepath):
    def translate_arff(data, meta):
        X = data[meta.names()[:-1]]  # everything but the last column
        # converts the record array to a normal numpy array. Posts a warning but SO says it's fine
        # see here for details: https://docs.scipy.org/doc/numpy-1.13.0/release.html
        X = X.view(np.float).reshape(data.shape + (-1,))
        y = data[meta.names()[-1]]
        y = y.view(np.float)
        return X, y
    data, meta = arff.loadarff(filepath)

    return translate_arff(data, meta)


def main():
    args = parse_args()
    train_X, train_y = load_arff_data(args.train)

    # Choose how many data points to use
    sample = train_X.shape[0]
    test_X, test_y = load_arff_data(args.test)

    mfr = MondrianForestRegressor(n_estimators=25, min_samples_split=10, n_jobs=4)

    pt = time.process_time()
    t = time.perf_counter()
    mfr.fit(train_X[:sample, :], train_y[:sample])
    process_time = time.process_time() - pt
    perf_time = time.perf_counter() - t

    print("Process time to train MF: {}".format(process_time))
    print("Perf time to train MF: {}".format(perf_time))

    y_mf, y_std = mfr.predict(test_X, return_std=True)
    print("MF prediction stats")
    print(stats.describe(y_mf))
    print("MF variance stats")
    print(stats.describe(y_std))
    print("MF MSE: {}".format(mean_squared_error(test_y, y_mf)))

    intervals = mfr.predict_interval(test_X, confidence=0.9)
    print("MF prediction interval size stats")
    print(stats.describe(intervals[:, 1] - intervals[:, 0]))

    correct_intervals = ((test_y >= intervals[:, 0]) & (test_y <= intervals[:, 1])).sum()
    print("Percentage correct intervals: {}".format(correct_intervals / len(test_y)))

    etr = ExtraTreesRegressor(n_estimators=25, min_samples_leaf=5, n_jobs=4)
    pt = time.process_time()
    t = time.perf_counter()
    etr.fit(train_X[:sample, :], train_y[:sample])
    process_time = time.process_time() - pt
    perf_time = time.perf_counter() - t
    print("\nProcess time to train ETR: {}".format(process_time))
    print("Perf time to train ETR: {}".format(perf_time))
    y_etr = etr.predict(test_X)
    print("ETR prediction stats")
    print(stats.describe(y_etr))
    print("ETR MSE: {}".format(mean_squared_error(test_y, y_etr)))


if __name__ == "__main__":
    main()
