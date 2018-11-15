"""
Parses the prediction files from MOA and skgarden output
and creates normalized metrics for the intervals.
"""
import argparse
import json
from pathlib import Path
from collections import OrderedDict, defaultdict
import re
import sys

import pandas as pd
import numpy as np
from natsort import natsorted
from tabulate import tabulate
from joblib import Parallel, delayed

from generate_figures import sort_nicely, gather_metric

MOA_METHODS = {"OnlineQRF", "OoBConformalRegressor", "OoBConformalApproximate", "PredictiveVarianceRF",
               "CPExact", "CPApproximate", "SGDQR"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="A dir containing sub-dirs of results, one per method."
                             "The sub-dirs contain csv files with output, one per dataset, per"
                             "experiment repeat."
                             "Sub-directory names will be used as method names in the plots.")
    parser.add_argument("--output", required=True,
                        help="The folder to create the output in")
    parser.add_argument("--only-pred-files", action="store_true", default=False,
                        help="When given, will not claculate metrics, only formatted prediction files."
                             " Use when you want RIS metrics for generate figures with large datasets.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="When given, will not check if the output folder already exists ,"
                             "potentially overwriting its contents.")
    parser.add_argument("--window-size",
                        help="The window size to use to calculate the per window metrics.")
    parser.add_argument("--expected-mer", type=float, default=0.1,
                        help="The expected MER for these experiments.")
    # parser.add_argument("--njobs", type=int, default=1,
    #                     help="Number of prediction files to process in parallel.")
    alg_selection = parser.add_mutually_exclusive_group()
    alg_selection.add_argument("--include-only", nargs='+',
                               help=" Include only the provided output directories")
    alg_selection.add_argument("--exclude", nargs='+',
                               help=" Exclude the provided output directories")
    parser.add_argument("--force-moa", action="store_true", default=False,
                        help="Enforce parsing of the dirs using the MOA format."
                             "Use when directory names don't match a method name (e.g. OnlineQRF),"
                             "MondrianForest parsing is used as the default in that case.")
    parser.add_argument("--njobs", help="The number of jobs to use for prediction file parsing/creation."
                                        "The default is to use all available cores.", type=int, default=-1)

    return parser.parse_args()


def parse_moa_line(line: str) -> str:
    # Expected line format: "Out 0: interval_low interval_high ,true_value"
    parsed_line = line[6:].replace(',', '').strip().replace(' ', ',')

    return parsed_line


def parse_skgarden_line(line: str) -> str:
    # Expected line format: "[interval_low interval_high], [true_value]"
    re_line = re.sub(r"[\[\],]", '', line).strip()
    parsed_line = re.sub(' +', ',', re_line)

    return parsed_line


def parse_file(filepath: Path, force_moa):
    """
    Reads a prediction file created either by MOA or skgarden_experiments,
    and creates a csv with the values, formatted as
    "interval_low,interval_high,true_value"
    The files are created under the same dir as the input .pred file, with
    the extension .pred.csv
    :param filepath: Path
        A Path object to a predictions file
    :param force_moa: When true will force MOA parsing to be used, regardless of containing directory name
    """
    if not filepath.with_suffix(".pred.csv").exists():
        # If the containing directory is one of the MOA methods, or we enforce it, use the MOA parser
        if filepath.parent.name in MOA_METHODS or force_moa:
            parse_line = parse_moa_line
        else:
            # Otherwise we assume it's a skgarden_experiments generated output file
            parse_line = parse_skgarden_line
        with filepath.open() as infile, filepath.with_suffix(".pred.csv").open('w') as outfile:
            outfile.write("interval_low,interval_high,true_value\n")
            for line in infile:
                parsed_line = parse_line(line)
                outfile.write(parsed_line + '\n')
        # TODO: Actually it seems like skgarden parsing works for both, pandas assumes the "Out: " is an index
        # TODO: Maybe include a sanity check here just to inform the user? The outcome is correct anyway


def create_pred_csvs(method_dir: Path, force_moa: bool, njobs):
    with Parallel(njobs) as parallel:
        parallel(delayed(parse_file)(res_file, force_moa)
                 for res_file in method_dir.glob("*.pred"))


def normalize(x, true_col):
    normalized = (x - min(true_col)) / (max(true_col) - min(true_col))
    return normalized


def gather_method_results(method_dir: Path):
    """
    Goes through all generated .pred.csv files in a method dir, collects the results
    per dataset, and creates a list of metric dataframes per metric.
    :param method_dir: A Path to method dir, contains repeats of experiments, with the suffix _x.pred.csv
    where x is the experiment repeat index.
    :return: Returns a dictionary {dataset_name: metric_df_list}
    """
    res = defaultdict(list)
    for res_file in method_dir.glob("*.pred.csv"):
        # Get rid of any suffixes in the filename, and the _X repeat indicator
        base_name = res_file.name.split('.')[0][:-2]

        df = pd.read_csv(res_file)
        df["interval_size"] = df["interval_high"] - df["interval_low"]
        true_max = df["true_value"].max()
        true_min = df["true_value"].min()
        df["relative_interval_size"] = df["interval_size"] / (true_max - true_min)
        df['error_rate'] = np.where(
            (df['true_value'] <= df['interval_high']) & (df['true_value'] >= df['interval_low']),
            0, 1)
        res[base_name].append(df)
    return res


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    assert output_path.parent != input_path, "Setting output path under input can cause issues, choose another path."
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    # Get all the directories under the input path
    method_dirs = [subpath for subpath in input_path.iterdir() if subpath.is_dir()]
    assert len(method_dirs) > 0, "There should be at least one directory under the input, found 0!".format(method_dirs)
    if args.exclude is not None:
        method_dirs = [subpath for subpath in method_dirs if subpath.name not in args.exclude]
    elif args.include_only is not None:
        method_dirs = [subpath for subpath in method_dirs if subpath.name in args.include_only]

    # Check if prediction file pre-processing has been done, otherwise do it

    # Gather the results for each method
    # Format: {method: {ds_name: measurement_df_list}}
    method_to_dsname_to_result_df_list = OrderedDict()
    sorted_dirs = sort_nicely(method_dirs)
    mean_tables = {}
    # TODO: For experiments with large outputs (prediction files with many lines) we have computationa and memory issues
    # The problem is that we maintain all results as a Pandas dataframe. For example for an experiment with 1M rows,
    # 10 repeats, 3 methods, the prediction data frames hold 90M float/double values. Processing these becomes
    # a challenge, we should find a way to 1) not store the complete data in memory 2) parallelize by method+metric
    for method_dir in sorted_dirs:
        # TODO: Have proper check that each result_X.csv file has respective result_X.pred
        if len(list(method_dir.glob("*.pred"))) == 0:
            raise FileNotFoundError("No prediction files found in {}!".format(method_dir))
        num_pred_files = len(list(method_dir.glob("*.pred")))
        num_processed_pred_files = len(list(method_dir.glob("*.pred.csv")))
        if num_processed_pred_files != num_pred_files:
            print(".pred.csv files missing in {}. Creating {}/{}".format(
                method_dir.name, num_pred_files - num_processed_pred_files, num_pred_files))
            create_pred_csvs(method_dir, args.force_moa)
        # After .pred.csv files have been created, gather metrics
        if not args.only_pred_files:
            method_to_dsname_to_result_df_list[method_dir.name] = gather_method_results(method_dir)

    if args.only_pred_files:
        print("Finished creating prediction files, exiting...")
        sys.exit()
    else:
        print("Prediction files created, continuing with metric calculation...")

    for metric in ["error_rate", "relative_interval_size"]:
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
            for dataset_name, metric_df in natsorted(ds_name_to_measurements.items()):
                if metric == "error_rate":
                    error_counts = metric_df.sum(axis=1)
                    error_rates = error_counts / metric_df.shape[1]
                    overall_mean = error_rates.mean()
                    overall_median_mean = error_rates.median()
                    overall_std_mean = error_rates.std()
                else:  # Then it's RIS
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

        def add_stats(df):
            df.loc['Mean'] = df.mean()
            df.loc['Median'] = df.median()
            df.loc['Std'] = df.std()
            return df

        def create_df(method_to_measurements_dict):
            dictionary = OrderedDict()
            dictionary["Dataset"] = ds_names
            dictionary.update(method_to_measurements_dict)
            df = pd.DataFrame(dictionary)
            df = df.set_index("Dataset")
            return add_stats(df)

        mean_aggregate_metric_df = create_df(method_to_mean_measurements)
        median_aggregate_metric_df = create_df(method_to_median_measurements)
        std_aggregate_metric_df = create_df(method_to_std_measurements)

        table_outpath = Path(args.output) / metric.replace(' ', '_')
        mean_aggregate_metric_df.to_csv(table_outpath.with_suffix(".means.csv"))
        median_aggregate_metric_df.to_csv(table_outpath.with_suffix(".medians.csv"))
        std_aggregate_metric_df.to_csv(table_outpath.with_suffix(".std.csv"))

        def create_table_str(df):
            float_format = ".3f" if metric == "error_rate" else ".2f"
            return tabulate(df, headers='keys', tablefmt='latex_booktabs', floatfmt=float_format)

        # Will try to rearrange columns in this order:
        # [MondrianForest, OnlineQRF, CPApproximate, CPExact].
        # This is the order used in the paper.
        order = sorted(mean_aggregate_metric_df.keys())
        # try:
        #     if "SGDQR" in method_ds_metric:
        #         order = ["SGDQR", "MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"]
        #     else:
        #         order = ["MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"]
        #     if args.exclude is not None:
        #         for excluded_method in args.exclude:
        #             order.remove(excluded_method)
        #     mean_aggregate_metric_df = mean_aggregate_metric_df[order]
        #     median_aggregate_metric_df = median_aggregate_metric_df[order]
        #     std_aggregate_metric_df = std_aggregate_metric_df[order]
        # except KeyError:
        #     # If a column was missing just leave them as they were
        #     pass
        mean_aggregate_metric_df = mean_aggregate_metric_df[order]
        median_aggregate_metric_df = median_aggregate_metric_df[order]
        std_aggregate_metric_df = std_aggregate_metric_df[order]

        mean_tables[metric] = mean_aggregate_metric_df
        # TODO: Create figures? Maybe window metric?
        mean_table_str = create_table_str(mean_aggregate_metric_df)
        median_table_str = create_table_str(median_aggregate_metric_df)
        std_table_str = create_table_str(std_aggregate_metric_df)

        table_outpath.with_suffix(".means.tex").write_text(mean_table_str)
        table_outpath.with_suffix(".medians.tex").write_text(median_table_str)
        table_outpath.with_suffix(".stds.tex").write_text(std_table_str)

    # Write json file with arguments to keep track of how output was generated
    json_file = output_path / "interval_metrics_settings.json"
    settings = vars(args)
    json_file.write_text(json.dumps(settings))

    # Ensure that the method dirs were not numbers, i.e. not repeats of confidence experiments.
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for method_name in method_dirs:
        if is_number(method_name.name):
            print("Seems like you're running this on top of confidence output, so we won't create utility output.")
            print("Method list was: {}.".format(method_dirs))
            print("Use confidence_utility_calculation.py instead!")
            sys.exit()

    # Create adjusted utility table with step time/utility function (using expected MER as deadline)

    # Compute utility as one minus the RIS
    mean_error_rates = mean_tables["error_rate"].iloc[:len(ds_names), ]  # Drop the aggregate stats rows
    utility = 1 - np.minimum(mean_tables["relative_interval_size"].iloc[:len(ds_names), ], 1)

    util_path = Path(args.output) / "utility"
    utility.to_csv(util_path.with_suffix(".csv"))
    util_table_str = create_table_str(utility)
    util_path.with_suffix(".tex").write_text(util_table_str)

    # Step function
    adjusted_utils = np.where(mean_error_rates > args.expected_mer, 0, utility)
    # adjusted_utils = tuf(utility, mean_error_rates, step)

    # Build up the adjusted util dataframe, add dataset names, aggregate stats
    adj_util_df = pd.DataFrame(adjusted_utils, columns=utility.columns.tolist())
    adj_util_df["Dataset"] = ds_names
    adj_util_df = add_stats(adj_util_df.set_index("Dataset"))

    adj_util_path = Path(args.output) / "adjusted_utility"
    adj_util_df.to_csv(adj_util_path.with_suffix(".csv"))
    ajd_util_table_str = create_table_str(adj_util_df)
    adj_util_path.with_suffix(".tex").write_text(ajd_util_table_str)


if __name__ == '__main__':
    main()
