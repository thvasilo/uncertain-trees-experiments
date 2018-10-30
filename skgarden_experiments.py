"""
Runs a number of experiments using skgarden MondrianForest.
The user provides a datadir which contains a number of arff files for regression, and a
prequential regression task is run on each one.

The output is one csv file per dataset, per experiment repeat.
Will also output two additional files per experiment:
<name>.time.csv
<name>.pred

These contain timing measurements and each individual prediction.

Usage: python skgarden_experiments.py --input path/to/data
"""
import argparse
from pathlib import Path
import json
from collections import namedtuple

import arff
import numpy as np
from skgarden import MondrianForestRegressor
import pandas as pd
from vowpalwabbit.sklearn_vw import VWRegressor

from evaluation_functions import mean_error_rate, mean_interval_size, prequential_interval_evaluation

from joblib import Parallel, delayed


class VWIntervalRegressor(object):
    def __init__(self, confidence):
        half_significance = (1.0 - confidence) / 2.0
        self.lower = VWRegressor(loss_function='quantile', quantile_tau=half_significance)
        self.upper = VWRegressor(loss_function='quantile', quantile_tau=1.0-half_significance)

    def partial_fit(self, X, y):
        self.lower.fit(X, y)
        self.upper.fit(X, y)

    def vw_predict_interval(self, X):
        lower = self.lower.predict(X)
        upper = self.upper.predict(X)
        return lower, upper

    def get_params(self):
        return {'lower': self.lower.get_params(), 'upper': self.upper.get_params()}

    def __getstate__(self):
        state = {}
        state['lower'] = dict(
            params=self.lower.get_params(), coefs=self.lower.get_coefs(), fit=self.lower.fit_)
        state['upper'] = dict(
            params=self.upper.get_params(), coefs=self.upper.get_coefs(), fit=self.upper.fit_)

        return state

    def __setstate__(self, state):
        self.lower.set_params(**state['lower']['params'])
        self.lower.set_coefs(state['lower']['coefs'])
        self.lower.fit_(state['lower']['fit'])

        self.upper.set_params(**state['upper']['params'])
        self.upper.set_coefs(state['upper']['coefs'])
        self.upper.fit_(state['upper']['fit'])


def load_arff_data(filepath):

    with open(str(filepath), 'r') as f:
        decoder = arff.ArffDecoder()
        d = decoder.decode(f, encode_nominal=True)
    # tvas: We are assuming the target/dependent is the last column
    data = np.array(d['data'])
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=['vw', 'mf'])
    parser.add_argument("--input", required=True,
                        help="Path to the folder containing the arff files.")
    parser.add_argument("--output", type=str,
                        help="Path to output folder. If not provided will create a dir named"
                             "MondrianForest under the input directory.")
    parser.add_argument("--n_estimators", type=int, default=10,
                        help="Number of trees to use.")
    parser.add_argument("--confidence", type=float, default=0.9,
                        help="Confidence level for intervals.")
    parser.add_argument("--window-size", type=int, default=1000,
                        help="Size of evaluation window.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of times to repeat each experiment")
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="When given, it will not check if the output folder exists already.")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Provide additional output in the console, 1 for per experiment progress, "
                             "2 to include per-window output")
    parser.add_argument("--njobs", type=int, default=1,
                        help="Number of repeat experiments to run in parallel")
    parser.add_argument("--no-additional-output", default=False, action="store_true",
                        help="When given, will not create predictions file and other computational metrics.")

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.input).absolute()
    if args.output is None:
        args.output = str(data_path / "MondrianForest")
    output_path = Path(args.output).absolute()
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    scorers = {"mean interval size": mean_interval_size,
               "mean error rate": mean_error_rate}

    learner_params_list = None
    if len(list(data_path.glob("*.arff"))) == 0:
        raise FileNotFoundError("Could not find any arff files under {}".format(data_path))

    for filepath in data_path.glob("*.arff"):
        X, y = load_arff_data(filepath)
        print("Running experiments on {}".format(filepath.name))
        # TODO: Nested parallelism, across files and repeats
        try:
            with Parallel(n_jobs=args.njobs, verbose=args.verbose) as parallel:
                learner_params_list = parallel(
                    delayed(run_experiment)(i, filepath, output_path, scorers, args, X, y)
                    for i in range(args.repeats))
        except:
            "Runtime error for dataset: {}".format(filepath.name)

    # Write experiment parameters to file
    if learner_params_list is None:
        raise Exception("No experiments performed, was the input dir correct?")
    results = {"arguments": vars(args), "learner_params": learner_params_list[0]}
    out_file = output_path / "settings.json"
    out_file.write_text(json.dumps(results))
    print("Output created under :{}".format(output_path))


def run_experiment(i, input_file, output_path, scorers, args, X, y):
    np.random.seed()
    window_size = args.window_size
    print("Running repeat {}/{}".format(i + 1, args.repeats))
    # Create and evaluate a regressor
    if args.algorithm == 'mf':
        regressor = MondrianForestRegressor(n_estimators=args.n_estimators)
    else:
        regressor = VWIntervalRegressor(args.confidence)

    # If asked to save predictions, create requisite file
    pred_path = output_path / (input_file.stem + "_{}.pred".format(i)) if not args.no_additional_output else None

    results = prequential_interval_evaluation(
        regressor, X, y, args.confidence, scorers, args.window_size, verbose=args.verbose,
        additional_output=pred_path)

    # Create index column
    num_windows = int(np.ceil(X.shape[0] / window_size))
    for score in scorers.keys():
        assert num_windows == len(results[score])
    window_index_list = list(range(window_size, (window_size * num_windows), window_size))
    # Last element of index is the dataset size, consistent with MOA
    window_index_list.append(X.shape[0])
    results["index"] = window_index_list

    # Save scores and index columns to csv
    included_columns = set(results.keys())
    # Create a df with only the score measurements and the index
    df = pd.DataFrame({k: results[k] for k in included_columns})
    df.to_csv(output_path / (input_file.stem + "_{}.csv".format(i)), index=False)
    return regressor.get_params()


if __name__ == "__main__":
    main()
