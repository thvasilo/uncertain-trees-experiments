import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required= True,
                        help="A dir containing csvs of results, one per repeat.")
    parser.add_argument("--output", required=True,
                        help="The folder to create the output in")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="When given, will not check if the output folder already exists ,"
                             "potentially overwriting its contents.")

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    assert output_path.parent != input_path, "Setting output path under input can cause issues, choose another path."
    output_path.mkdir(parents=True, exist_ok=args.overwrite)
    computational_metrics = ['evaluation time (cpu seconds)',
                             'model cost (RAM-Hours)',
                             'model serialized size (bytes)']

    combined_df = pd.DataFrame(columns=computational_metrics)
    prev_means = None
    experiment_values = {}
    # TODO: Support for multiple method directories, creation of comparison tables
    experiment_index = 0
    for experiment_file in input_path.glob("*.csv"):
        if experiment_file.match("*.pred.csv") or experiment_file.match("*.time.csv"):
            continue
        experiment_value = experiment_file.stem
        experiment_values[experiment_index] = experiment_value
        df = pd.read_csv(experiment_file)
        # Get the last line of the dataframe, we want the final measurements of these metrics
        df = df.iloc[[-1]]
        metrics_df = df[computational_metrics]
        # means = metrics_df.mean()
        # means_df = means.to_frame().T.rename({0: experiment_value}, axis='index')
        combined_df = combined_df.append(metrics_df, ignore_index=True)
        experiment_index += 1

    # Calculate the means for each dataset/experiment
    combined_df = combined_df.rename(experiment_values, axis='index').sort_index(ascending=True)

    def include_aggregates(df):
        df.loc['Mean'] = df.mean()
        df.loc['Median'] = df.median()
        df.loc['Std'] = df.std()
        return df

    combined_df = include_aggregates(combined_df)

    def create_table_str(df):
        #float_format = [".2f", ".4f", ".2f"]  # if outpath.name == "mean_error_rate" else ".2f"
        return tabulate(df, headers='keys', tablefmt='latex_booktabs')
    combined_df.to_csv(output_path / "averaged_comp_metrics.csv")
    table_str = create_table_str(combined_df)
    table_file = output_path / "averaged_comp_metrics.tex"
    table_file.write_text(table_str)


if __name__ == '__main__':
    main()