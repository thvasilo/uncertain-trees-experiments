import argparse
from pathlib import Path
import json

import arff
import numpy as np
from skgarden import MondrianForestRegressor
import pandas as pd

from scikit_online_eval.interval_metrics import IntervalScorer, mean_error_rate, mean_interval_size
from scikit_online_eval.evaluation_functions import prequential_evaluation

from joblib import Parallel, delayed


def load_arff_data(filepath):

    with open(filepath, 'r') as f:
        decoder = arff.ArffDecoder()
        d = decoder.decode(f, encode_nominal=True)
    # tvas: We are assuming the target/dependent is the last column
    data = np.array(d['data'])
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder",
                        help="Path to the folder containing the arff files.")
    parser.add_argument("--output", type=str,
                        help="Path to output folder. If not provided will create a dir named"
                             "MondrianForest")
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
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Provide additional output in the console")
    parser.add_argument("--njobs", type=int, default=1,
                        help="Number of experiments to run in parallel")

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_folder)
    if args.output is None:
        args.output = str(data_path / "MondrianForest")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    # TODO: Other/customizable scorers?
    interval_size_scorer = IntervalScorer(
        mean_interval_size, {"confidence": args.confidence})
    error_rate_scorer = IntervalScorer(
        mean_error_rate, {"confidence": args.confidence})

    scorers = {"mean interval size": interval_size_scorer,
               "mean error rate": error_rate_scorer}

    for filepath in data_path.glob("*.arff"):
        X, y = load_arff_data(filepath)
        print("Running experiments on {}".format(filepath.name))

        with Parallel(n_jobs=args.njobs) as parallel:
            learner_params_list = parallel(
                delayed(run_experiment)(i, filepath, output_path, scorers, args, X, y)
                for i in range(args.repeats))

    # Write experiment parameters to file
    results = {"arguments": vars(args), "learner_params": learner_params_list[0]}
    out_file = output_path / "arguments.json"
    out_file.write_text(json.dumps(results))


def run_experiment(i, filepath, output_path, scorers, args, X, y):
    window_size = args.window_size
    print("Running repeat {}/{}".format(i + 1, args.repeats))
    # Create and evaluate an MF regressor
    mfr = MondrianForestRegressor(n_estimators=args.n_estimators, verbose=args.verbose)
    results = prequential_evaluation(mfr, X, y, scorers, args.window_size)
    results["learner_params"] = mfr.get_params()

    # Create index column
    num_windows = int(np.ceil(X.shape[0] / window_size))
    for score in scorers.keys():
        assert num_windows == len(results[score])
    window_index_list = list(range(window_size, (window_size * num_windows), window_size))
    # Last element of index is the dataset size, as MOA does
    window_index_list.append(X.shape[0])
    results["index"] = window_index_list

    # Save scores and index columns to csv
    included_columns = set(scorers.keys())
    included_columns.add("index")
    # Create a df with only the score measurements and the index
    df = pd.DataFrame({k: results[k] for k in included_columns})
    df.to_csv(output_path / (filepath.stem + "_{}.csv".format(i)), index=False)
    return mfr.get_params()


if __name__ == "__main__":
    main()
