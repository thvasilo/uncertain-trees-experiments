import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import numpy as np
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="A dir containing output from interval_metrics.py")
    parser.add_argument("--output", default="",
                        help="The folder to create the output in")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="When given, will not check if the output folder already exists ,"
                             "potentially overwriting its contents.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inpath = Path(args.input)
    outpath = Path(args.output if args.output != "" else args.input) / "adjusted_utility"
    # Read in the error and ris csv's
    error_rate_means = pd.read_csv(inpath / "error_rate.means.csv")\
        .set_index("Dataset").iloc[:-3, :] # Drop aggregate stats rows
    ris = pd.read_csv(inpath / "relative_interval_size.means.csv")\
        .set_index("Dataset").iloc[:-3, :]

    # Extract the percentages from the column titles and subtract the expected confidence from 1 to get significance
    cols = error_rate_means.columns.tolist()
    significances = [1 - float(col) for col in cols]

    # For each column in the error df, use the respective significance and ris to calculate the adj utility
    util_dict = OrderedDict()

    # (Some) code duplication from interval_metrics.py
    def dropoff(util, err, expected_mer, func):
        if err > expected_mer:
            return max(func(util, err, expected_mer), 0)
        else:
            return max(util, 0)

    # Vectorize the function
    tuf = np.vectorize(dropoff)

    # Linear dropoff, utility is zero when MER is twice bigger than expected
    def linear(util, error, expected_mer):
        return max(util, 0) * max((2 - (error / expected_mer)), 0)

    for column, significance in zip(error_rate_means, significances):
        ris_col = ris[column]
        error_col = error_rate_means[column]
        util_col = 1 - ris_col
        weight_col = np.maximum((2 - (error_col / significance)), 0)
        util_dict[column] = tuf(util_col, error_col, significance, linear)


    def add_stats(df):
        df.loc['Mean'] = df.mean()
        df.loc['Median'] = df.median()
        df.loc['Std'] = df.std()
        return df

    util_df = add_stats(pd.DataFrame(util_dict, index=ris.index, columns=cols))

    util_df.to_csv(outpath.with_suffix(".csv"))

    def create_table_str(df):
        float_format = ".2f"
        return tabulate(df, headers='keys', tablefmt='latex_booktabs', floatfmt=float_format)

    util_table_str = create_table_str(util_df)
    outpath.with_suffix(".tex").write_text(util_table_str)
