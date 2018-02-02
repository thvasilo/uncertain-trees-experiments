"""
Parses the prediction files from MOA and skgarden output
and creates normalized metrics for the intervals.
"""
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict
import re

import pandas as pd
import numpy as np
from natsort import natsorted
from tabulate import tabulate

from generate_figures import sort_nicely, gather_metric

MOA_METHODS = {"OnlineQRF", "OoBConformalRegressor", "OoBConformalApproximate", "PredictiveVarianceRF"}


def parse_moa_line(line: str) -> str:
    # Expected line format: "Out 0: interval_low interval_high ,true_value"
    parsed_line = line[6:].replace(',', '').strip().replace(' ', ',')

    return parsed_line


def parse_skgarden_line(line: str) -> str:
    # Expected line format: "[interval_low interval_high], [true_value]"
    re_line = re.sub(r"\[|\]|,", '', line).strip()
    parsed_line = re.sub(' +', ',', re_line)

    return parsed_line


def parse_file(filepath: Path, parse_line):
    """
    Reads a prediction file created either by MOA or skgarden_experiments,
    and creates a csv with the values, formatted as
    "interval_low,interval_high,true_value"
    The files are created under the same dir as the input .pred file, with
    the extension .pred.csv
    :param filepath: Path
        A Path object to a predictions file
    :param parse_line: str -> str
        A callable that takes a string, parses it, and returns a string in
        the expected format.
    """

    if filepath.with_suffix(".pred.csv").exists():
        print("{} already exists, skipping...".format(filepath.with_suffix(".pred.csv")))
    else:
        with filepath.open() as infile, filepath.with_suffix(".pred.csv").open('w') as outfile:
            outfile.write("interval_low,interval_high,true_value\n")
            for line in infile:
                parsed_line = parse_line(line)
                outfile.write(parsed_line + '\n')


def create_pred_csvs(method_dir: Path, force_moa: bool):
    for res_file in method_dir.glob("*.pred"):
        if res_file.parent.name in MOA_METHODS or force_moa:
            parse_file(res_file, parse_moa_line)
        else:
            parse_file(res_file, parse_skgarden_line)

def normalize(x, true_col):
    normalized = (x - min(true_col)) / (max(true_col) - min(true_col))
    return normalized


def gather_method_results(method_dir: Path):
    res = defaultdict(list)
    for res_file in method_dir.glob("*.pred.csv"):
        # Get rid of any suffixes in the filename, and the _X repeat indicator
        base_name = res_file.name.split('.')[0][:-2]

        df = pd.read_csv(res_file)
        df["interval_size"] = df["interval_high"] - df["interval_low"]
        true_max = df["true_value"].max()
        true_min = df["true_value"].min()
        df["relative_interval_size"] = df["interval_size"] / (true_max - true_min)
        df['correct'] = np.where((df['interval_high'] >= df['true_value']) & (df['interval_low'] <= df['true_value']),
                                 df['relative_interval_size'], np.nan)
        # df["interval_high_norm"] = normalize(df["interval_high"], df["true_value"])
        # df["interval_lower_norm"] = normalize(df["interval_low"], df["true_value"])
        res[base_name].append(df)
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="A dir containing sub-dirs of results, one per method."
                             "The sub-dirs contain csv files with output, one per dataset, per"
                             "experiment repeat."
                             "Sub-directory names will be used as method names in the plots.")
    parser.add_argument("--output", required=True,
                        help="The folder to create the output in")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="When given, will not check if the output folder already exists ,"
                             "potentially overwriting its contents.")
    parser.add_argument("--window-size",
                        help="The window size to use to calculate the per window metrics.")
    alg_selection = parser.add_mutually_exclusive_group()
    alg_selection.add_argument("--include-only", nargs='+',
                               help=" Include only the provided output directories")
    alg_selection.add_argument("--exclude", nargs='+',
                               help=" Exclude the provided output directories")
    parser.add_argument("--force-moa", action="store_true", default=False,
                        help="Enforce parsing of the dirs using the MOA format."
                             "Use when directory names don't match a method name (e.g. OnlineQRF),"
                             "MondrianForest parsing is used as the default in that case.")

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    assert output_path.parent != input_path, "Setting output path under input can cause issues, choose another path."
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    # Get all the directories under the input path
    method_dirs = [subpath for subpath in input_path.iterdir() if subpath.is_dir()]
    if args.exclude is not None:
        method_dirs = [subpath for subpath in method_dirs if subpath.name not in args.exclude]
    elif args.include_only is not None:
        method_dirs = [subpath for subpath in method_dirs if subpath.name in args.include_only]

    # Check if prediction file pre-processing has been done, otherwise do it

    # Gather the results for each method
    # Format: {method: {ds_name: measurement_df_list}}
    method_to_dsname_to_result_df_list = OrderedDict()
    sorted_dirs = sort_nicely(method_dirs)
    for method_dir in sorted_dirs:
        # TODO: Have proper check that each result_X.csv file has respective result_X.pred
        if len(list(method_dir.glob("*.pred"))) == 0:
            raise FileNotFoundError("No prediction files found in {}!".format(method_dir))
        if len(list(method_dir.glob("*.pred.csv"))) != len(list(method_dir.glob("*.pred"))):
            create_pred_csvs(method_dir, args.force_moa)
        method_to_dsname_to_result_df_list[method_dir.name] = gather_method_results(method_dir)

    for metric in ["correct", "relative_interval_size"]:
        method_ds_metric = OrderedDict()
        for method, ds_to_measurements in method_to_dsname_to_result_df_list.items():
            # TODO: Make it possible to iterate over metrics?
            method_ds_metric[method] = gather_metric(ds_to_measurements, metric)

        all_names = []
        method_to_mean_measurements = OrderedDict()
        method_to_median_measurements = OrderedDict()
        method_to_std_measurements = OrderedDict()
        for method, ds_name_to_measurements in method_ds_metric.items():
            ds_names = []
            ds_means = []
            ds_medians = []
            ds_stds = []
            # Get the measurements for the requested data, and calc their stats
            for dataset_name, metric_df in natsorted(ds_name_to_measurements .items()):
                if metric == "correct":
                    relative_interval_sums = metric_df.sum(axis=1)  # Mean for each example over repeats
                    non_na_counts = metric_df.count(axis=1)
                    # Correctness is metric that tries to measure how good a method is, by multiplying its
                    # relative intervals by the number of mistakes it does.
                    # So methods that are correct often but have relatively large intervals should fare
                    # better than methods that have small intervals, but are incorrect often.
                    # correctness = relative_interval_sums / (counts / metric_df.shape[1])
                    correctness = non_na_counts / metric_df.shape[1]
                    overall_mean = correctness.mean()
                    overall_median_mean = correctness.median()
                    overall_std_mean = correctness.std()
                else:
                    relative_interval_means = metric_df.mean()  # Mean for each example over repeats
                    relative_interval_medians = metric_df.median()  # Median for each example over repeats
                    relative_interval_stds = metric_df.std()  # Std for each example over repeats
                    overall_mean = relative_interval_means.mean()
                    overall_median_mean = relative_interval_medians.mean()
                    overall_std_mean = relative_interval_stds.mean()

                ds_means.append(overall_mean)
                ds_medians.append(overall_median_mean)
                ds_stds.append(overall_std_mean)

                ds_names.append(dataset_name)

            all_names.append(ds_names)

            method_to_mean_measurements[method] = ds_means
            method_to_median_measurements[method] = ds_medians
            method_to_std_measurements[method] = ds_stds

        # Assert datasets are in correct order between methods
        prev_names = []
        for ds_names in all_names:
            if len(prev_names) == 0:
                prev_names = ds_names
                continue
            for left_name, right_name in zip(prev_names, ds_names):
                assert left_name == right_name, \
                    "DS order mismatch: {}, {}".format(left_name, right_name)

        def create_df(method_to_measurements_dict):
            dictionary = OrderedDict()
            dictionary["Dataset"] = ds_names
            dictionary.update(method_to_measurements_dict)
            df = pd.DataFrame(dictionary)
            df = df.set_index("Dataset")
            df.loc['Mean'] = df.mean()
            df.loc['Median'] = df.median()
            df.loc['Std'] = df.std()
            return df

        mean_aggregate_metric_df = create_df(method_to_mean_measurements)
        median_aggregate_metric_df = create_df(method_to_median_measurements)
        std_aggregate_metric_df = create_df(method_to_std_measurements)

        table_outpath = Path(args.output) / metric.replace(' ', '_')
        mean_aggregate_metric_df.to_csv(table_outpath .with_suffix(".means.csv"))
        median_aggregate_metric_df.to_csv(table_outpath .with_suffix(".medians.csv"))
        std_aggregate_metric_df.to_csv(table_outpath .with_suffix(".std.csv"))

        def create_table_str(df):
            float_format = ".2f"
            return tabulate(df, headers='keys', tablefmt='latex_booktabs', floatfmt=float_format)

        # TODO: Create figures? Maybe window metric?
        mean_table_str = create_table_str(mean_aggregate_metric_df)
        median_table_str = create_table_str(median_aggregate_metric_df)
        std_table_str = create_table_str(std_aggregate_metric_df)

        table_outpath.with_suffix(".means.tex").write_text(mean_table_str)
        table_outpath.with_suffix(".medians.tex").write_text(median_table_str)
        table_outpath.with_suffix(".stds.tex").write_text(std_table_str)


if __name__ == '__main__':
    main()