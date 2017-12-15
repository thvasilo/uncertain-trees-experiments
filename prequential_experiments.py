import argparse
from pathlib import Path
import json

import arff
import numpy as np
from skgarden import MondrianForestRegressor

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

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_folder)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True) # Is it OK if the path exists? should be arg to avoid overwriting (default false)

    interval_size_scorer = IntervalScorer(
        mean_interval_size, {"confidence": args.confidence})
    error_rate_scorer = IntervalScorer(
        mean_error_rate, {"confidence": args.confidence})

    scorers = {"mean_interval_size": interval_size_scorer,
               "mean_error_rate": error_rate_scorer}

    for filepath in data_path.glob("*.arff"):
        X, y = load_arff_data(filepath)
        print(X.shape)
        for i in range(args.repeats):
            mfr = MondrianForestRegressor(n_estimators=args.n_estimators)
            results = prequential_evaluation(mfr, X, y, scorers, args.window_size)
            out_file = output_path / (filepath.stem + "_{}.json".format(i))
            results["arguments"] = vars(args)
            results["learner_params"] = mfr.get_params()
            out_file.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
