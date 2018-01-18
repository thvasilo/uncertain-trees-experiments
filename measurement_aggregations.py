"""
Creates aggregations over datasets from the per method/dataset results
created in generate_figures.py in the create_tables method.

Provided a directory created with the above method will create
additional output in the same directory with informative measures
over all datasets for which we have results.
"""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True,
                        help="The directory containing the input files."
                             "Must have been created with generate_figures"
                             " --create-tables.")
    parser.add_argument("--expected-error", default=0.1,
                        help="The expected error level for the experiment."
                             "Used to calculate the avg deviation from it.")

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    mer_file = input_path / "mean_error_rate.means.csv"

    mer_df = pd.read_csv(mer_file)

    # For each metric value, take its absolute difference with the requested confidence
    abs_mer_diff_df = mer_df.drop("Dataset", axis=1).apply(lambda x: abs(x - args.expected_error), axis=1)

    mean_abs_mer_diff = abs_mer_diff_df.mean()
    std_abs_mer_diff = abs_mer_diff_df.std()

    mean_abs_mer_diff.to_csv(input_path / "mean_abs_mer_diff.csv")
    std_abs_mer_diff.to_csv(input_path / "std_abs_mer_diff.csv")


if __name__ == "__main__":
    main()