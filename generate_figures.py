import argparse
from collections import defaultdict
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import pandas as pd
from pylab import *

plt.style.use(['seaborn-whitegrid'])

rcParams = matplotlib.rcParams
params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,  # Maybe change to Tex for final figures
   'figure.figsize': [6, 6]
   }
rcParams.update(params)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        help="A dir containing dis of results, one per method."
                             "Sub-directory names will be used as method names in the plots")
    parser.add_argument("--output",
                        help="The folder to create the output in")
    parser.add_argument("--metric",
                        help="The metric to plot, should correspond to csv output")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="When given, will not check if the output folder exists already.")

    return parser.parse_args()


def gather_method_results(res_path: Path):
    """Returns a dict from dataset name to a list of result dataframes"""
    res = defaultdict(list)
    for res_file in res_path.glob("*.csv"):
        base_name = res_file.name[:-6]
        res[base_name].append(pd.read_csv(res_file))
    return res


def gather_metric(results_dict, metric):
    """
    Returns a dict from dataset name to a df
    containing the aggregated measurements for the requested metric
    :param results_dict: {dataset_name: list_of_result_dataframes}
        The list has one dataframe per experiment, with all the metrics.
    :param metric: String, name of metric (common across dataframes)
    :return: dict {dataset_name: dataframe_of_metric}
        The df has one line per experiment
    """
    dataset_to_metric = {}
    for ds_name, df_list in results_dict.items():
        metric_df = pd.DataFrame()
        for inner_df in df_list:
            metric_df = metric_df.append(inner_df[metric])
        dataset_to_metric[ds_name] = metric_df
    return dataset_to_metric


def plot_metric(method_metric_dict, dataset_name, x_axis, metric_name):
    """Returns an error-bar plot of the aggregated statistics (mean, var) for
        a specific metric, for each method in the provided dict.
    :param method_metric_dict: {method: {ds_name: measurements_df}}
        Maps each method to dataset names, each one with one aggregate
        measurements df.
    :param dataset_name: The specific dataset name we want to plot.
    :param x_axis: The x axis that corresponds to the y measurements.
    :param metric_name: The name of the metric we are plotting
    :return: A matplotlib Axes object, containing the plotted figure (no legends)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Instance')
    ax.set_ylabel(metric_name.title())
    marker = itertools.cycle(('o', '^', 's', 'x', '*'))
    # Get the method name, and its measurements for all datasets
    for method, ds_metric_dict in method_metric_dict.items():
        # Get the measurements for the requested data, and calc their stats
        metric_df = ds_metric_dict[dataset_name]
        mu = metric_df.mean()
        std_dev = metric_df.std()
        # Plot the line for the method and dataset
        ax.errorbar(x_axis, mu, std_dev, label=method, marker=next(marker),
                    capsize=3)
    return ax


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    assert output_path.parent != input_path, "Setting output path under input can cause issues, choose another path."
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    # Get all the directories under the input path
    method_dirs = [subpath for subpath in input_path.iterdir() if subpath.is_dir()]

    # Gather the results for each method
    # Format: {method: {ds_name: measurement_df_list}}
    method_to_dsname_to_result_df_list = {}
    for method_dir in method_dirs:
        method_to_dsname_to_result_df_list[method_dir.name] = gather_method_results(method_dir)

    # Aggregate the list of result df to a single df per dataset, per method.
    # Format: {method: {ds_name: measurements_df}}
    # Each line in measurements_df is one experiment
    method_ds_measure = {}
    for method, ds_to_measurements in method_to_dsname_to_result_df_list.items():
        # TODO: Make it possible to iterate over metrics?
        method_ds_measure[method] = gather_metric(ds_to_measurements, args.metric)

    # Gather the names of datasets
    ds_names = set()
    for method, ds_name_to_measure in method_ds_measure.items():
        ds_names.update(ds_name_to_measure.keys())

    # All methods should have same x_axis, so just choose one
    sample_method = list(method_to_dsname_to_result_df_list.keys())[0]
    # Create and save one figure per dataset
    for dataset in ds_names:
        # From one of the methods, for this dataset, first experiment, get the index, as ints
        try:
            # If this doesn't work, we don't have MOA generated experiments
            x_axis = method_to_dsname_to_result_df_list[sample_method][dataset][0]["learning evaluation instances"].astype(int)
        except KeyError:
            # In which case we should have only Python-generated experiments, which should have an index column
            x_axis = method_to_dsname_to_result_df_list[sample_method][dataset][0]["index"].astype(int)
        ax = plot_metric(method_ds_measure, dataset, x_axis, args.metric)
        plt.legend()
        outpath = Path(args.output) / (dataset + "-" + args.metric + ".pdf")
        plt.savefig(str(outpath))


if __name__ == "__main__":
    main()
