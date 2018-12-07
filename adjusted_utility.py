import argparse
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input directory, should contain utility.csv file"
                                                       "created using interval_metrics.py")
    parser.add_argument("--expected-mer", type=float, default=0.1)
    parser.add_argument("--table-order", nargs='+')
    alg_selection = parser.add_mutually_exclusive_group()
    alg_selection.add_argument("--include-only", nargs='+',
                               help="Include only the provided output directories")
    alg_selection.add_argument("--exclude", nargs='+',
                               help="Exclude the provided output directories")

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    utilities = pd.read_csv(input_path / "utility.csv", index_col=0)
    # Read in mean error rates, exclude final 3 lines that are aggregate statistics
    error_rates = pd.read_csv(input_path / "error_rate.means.csv", index_col=0).iloc[:-3, ]

    if args.exclude is not None:
        try:
            utilities = utilities.drop(args.exclude, axis=1)
            error_rates = error_rates.drop(args.exclude, axis=1)
        except ValueError as e:
            logging.warning("Could not drop requested columns,  error was \"{}\"".format(e))
    elif args.include_only is not None:
        try:
            utilities = utilities[args.include_only]
            error_rates = error_rates[args.include_only]
        except KeyError as e:
            logging.warning("Could not include requested columns,  error was \"{}\"".format(e))

    # We set the half life to be at an error rate of 1.5 times the requested one
    # That means that the utility of a method will be half if the method has 1.5 times the requested error rate
    half_life_deviation = 0.5 * args.expected_mer
    decay_lambda = np.log(2) / half_life_deviation

    # If the error rate is not smaller/equal to expected, multiply utility by decay rate determined by error deviation
    adjusted_utilities = utilities.where(error_rates <= args.expected_mer,
                                         np.multiply(utilities,
                                                     np.exp(-decay_lambda * (error_rates - args.expected_mer))))

    try:
        adjusted_utilities = adjusted_utilities[args.table_order]
    except KeyError:
        logging.warning("Could not set column order to {},  available columns were {}".format(
            args.table_order, adjusted_utilities.columns.values))
        pass

    # Write json file with arguments to keep track of how output was generated
    json_file = input_path/ "adjusted_utility_settings.json"
    settings = vars(args)
    json_file.write_text(json.dumps(settings))

    def add_stats(df: pd.DataFrame) -> pd.DataFrame:
        df.loc['Mean'] = df.mean()
        df.loc['Median'] = df.median()
        df.loc['Std'] = df.std()
        return df

    adjusted_utilities = add_stats(adjusted_utilities)

    def create_table_str(df):
        float_format = ".2f"
        return tabulate(df, headers='keys', tablefmt='latex_booktabs', floatfmt=float_format)

    adj_util_path = input_path / "exp_adjusted_utility"
    adjusted_utilities.to_csv(adj_util_path.with_suffix(".csv"))
    ajd_util_table_str = create_table_str(adjusted_utilities)
    adj_util_path.with_suffix(".tex").write_text(ajd_util_table_str)


if __name__ == '__main__':
    main()
