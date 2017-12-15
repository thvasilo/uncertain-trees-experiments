import argparse
from pathlib import Path
import json

import arff
import numpy as np
from skgarden import MondrianForestRegressor
import pandas as pd

from scikit_online_eval.interval_metrics import IntervalScorer, mean_error_rate, mean_interval_size
from scikit_online_eval.evaluation_functions import prequential_evaluation


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
    parser.add_argument("--output",
                        help="Path to output folder.")
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

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_folder)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    interval_size_scorer = IntervalScorer(
        mean_interval_size, {"confidence": args.confidence})
    error_rate_scorer = IntervalScorer(
        mean_error_rate, {"confidence": args.confidence})

    scorers = {"mean_interval_size": interval_size_scorer,
               "mean_error_rate": error_rate_scorer}

    window_size = args.window_size
    for filepath in data_path.glob("*.arff"):
        X, y = load_arff_data(filepath)
        print(X.shape)
        for i in range(args.repeats):
            # Create and evaluate an MF regressor
            mfr = MondrianForestRegressor(n_estimators=args.n_estimators)
            results = prequential_evaluation(mfr, X, y, scorers, window_size)
            out_file = output_path / (filepath.stem + "_{}.json".format(i))
            # Build up results
            results["arguments"] = vars(args)
            results["learner_params"] = mfr.get_params()
            results["filename"] = filepath.name

            num_windows = int(np.ceil(X.shape[0] / window_size))
            for score in scorers.keys():
                assert num_windows == len(results[score])
            # tvas: window_index_list won't be correct for trailing windows, should have the last element be length(X)
            # i.e. if data set has 9500 elements, and window_size is 1000, window_index_list[-1] will be
            # 10000, but should be 9500.
            window_index_list = list(range(window_size, (window_size * num_windows) + 1, window_size))
            results["index"] = window_index_list
            out_file.write_text(json.dumps(results))
            # Save scores and index to csv
            include = set(scorers.keys())
            include.add("index")
            df = pd.DataFrame({k: results[k] for k in include})
            df.to_csv(output_path / (filepath.stem + "_{}.csv".format(i)), index=False)


if __name__ == "__main__":
    main()
