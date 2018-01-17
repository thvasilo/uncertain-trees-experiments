"""
Creates figures for a specific metric from a directory containing subdirs of result csv files.
Will aggregate all the results for every dataset, and plot its mean
value along with its std across experiments.

Used to plot the output from the skgarden_experiments and moa_experiments scripts.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from tabulate import tabulate

plt.style.use(['seaborn-whitegrid'])

rcParams = matplotlib.rcParams


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required= True,
                        help="A dir containing sub-dirs of results, one per method."
                             "The sub-dirs contain csv files with output, one per dataset, per"
                             "experiment repeat."   
                             "Sub-directory names will be used as method names in the plots.")
    parser.add_argument("--output", required= True,
                        help="The folder to create the output in")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="When given, will not check if the output folder already exists ,"
                             "potentially overwriting its contents.")
    parser.add_argument("--create-tables", action="store_true", default=False,
                        help="When given, will create a table with the mean values over"
                             "all windows aggregated for each dataset")
    parser.add_argument("--fig-height", type=int, default=6)
    parser.add_argument("--fig-width", type=int, default=6)

    return parser.parse_args()


def gather_method_results(res_path: Path):
    """ Reads csv files into lists of pandas dfs.
    :param res_path: Path
        A Path object to the location of the result csv's.
        The expected filenames are formatted as dataset_name_X.csv,
        where X is the experiment repeat number.
    :return: A dict from dataset name to a list of result dataframes
    """
    res = defaultdict(list)
    for res_file in res_path.glob("*.csv"):
        base_name = res_file.name[:-6]  # _X.csv is 6 characters
        res[base_name].append(pd.read_csv(res_file))
    return res


def gather_metric(results_dict, metric):
    """
    Returns a dict from dataset name to a df containing the aggregated
    measurements for the requested metric.
    :param results_dict: {dataset_name: list_of_result_dataframes}
        The list has one dataframe per experiment, with all the metrics.
        It corresponds to the output of gather_method_results().
    :param metric: String, name of metric (common across dataframes)
    :return: dict {dataset_name: dataframe_of_metric}
        The df has one line per experiment, each column is the measurement
        for one window.
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
       a specific dataset and metric, for each method in the provided dict.
    :param method_metric_dict: {method: {ds_name: measurements_df}}
        Maps each method to dataset names, each one with one aggregate
        measurements df.
        The df has one line per experiment, each column is the measurement
        for one window.
    :param dataset_name: The specific dataset name we want to plot.
    :param x_axis: The x axis values tha correspond to the y measurements.
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
        # ax.errorbar(x_axis, mu, std_dev, label=method, marker=next(marker),
        #             capsize=3)
        ax.plot(x_axis, mu, label=method, marker=next(marker))
        ax.fill_between(x_axis, mu - std_dev, mu + std_dev, alpha=.25, linewidth=0)
    return ax


def create_metric_table(method_metric_dict, outpath):
    """Creates a Latex booktabs table for a specific dataset and metric, for
    each method in the provided dict as well as a csv representation,
    and writes them both to disk.
    :param method_metric_dict: {method: {ds_name: measurements_df}}
        Maps each method to dataset names, each one with one aggregate
        measurements df.
        The df has one line per experiment, each column is the measurement
        for one window.
    :param outpath: Path
        A Path object to the output tex file we will create.
    """

    method_to_measurements = {}
    method_to_std_measurements = {}
    ds_names = []
    for method, ds_metric_dict in method_metric_dict.items():
        ds_means = []
        ds_stds = []
        # Get the measurements for the requested data, and calc their stats
        for dataset_name, metric_df in ds_metric_dict.items():
            window_mean = metric_df.mean()
            window_std = metric_df.std()
            overall_mean = window_mean.mean()
            overall_std = window_std.mean()
            ds_names.append(dataset_name)
            ds_means.append(overall_mean)
            ds_stds.append(overall_std)

        method_to_measurements[method] = ds_means
        method_to_std_measurements[method] = ds_stds

    # TODO: Assert datasets are in correct order between methods

    pd_dict = {"Dataset": ds_names}
    pd_dict.update(method_to_measurements)
    aggregate_metric_df = pd.DataFrame(pd_dict)

    std_pd_dict = {"Dataset": ds_names}
    std_pd_dict.update(method_to_std_measurements)
    std_aggregate_metric_df = pd.DataFrame(std_pd_dict)

    aggregate_metric_df.to_csv(outpath.with_suffix("_means.csv"), index=False)
    std_aggregate_metric_df.to_csv(outpath.with_suffix("_stds.csv"), index=False)

    table_str = tabulate(aggregate_metric_df, headers='keys', tablefmt='latex_booktabs', showindex=False)
    std_table_str = tabulate(std_aggregate_metric_df, headers='keys', tablefmt='latex_booktabs', showindex=False)

    outpath.with_suffix(".tex").write_text(table_str)
    outpath.with_suffix(".tex").write_text(std_table_str)

def main():
    args = parse_args()

    params = {
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,  # Maybe change to Tex for final figures
        'figure.figsize': [args.fig_width, args.fig_height]
    }
    rcParams.update(params)

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

    for metric in ["mean error rate", "mean interval size"]:
        # Aggregate the list of result df to a single df per dataset, per method.
        # Format: {method: {ds_name: measurements_df}}
        # Each line in measurements_df is one experiment
        method_ds_measure = {}
        for method, ds_to_measurements in method_to_dsname_to_result_df_list.items():
            # TODO: Make it possible to iterate over metrics?
            method_ds_measure[method] = gather_metric(ds_to_measurements, metric)

        # Gather the names of datasets
        ds_names = set()
        for method, ds_name_to_measure in method_ds_measure.items():
            ds_names.update(ds_name_to_measure.keys())

        # All methods should have same x_axis, so just choose one
        sample_method = list(method_to_dsname_to_result_df_list.keys())[0]

        if args.create_tables:
            table_outpath = Path(args.output) / metric
            create_metric_table(method_ds_measure, table_outpath)

        # Create and save one figure per dataset
        for dataset in ds_names:
            # From one of the methods, for this dataset, first experiment, get the index, as ints
            try:
                # If this doesn't work, we don't have MOA generated experiments
                x_axis = method_to_dsname_to_result_df_list[sample_method][dataset][0]["learning evaluation instances"].astype(int)
            except KeyError:
                # In which case we should have only Python-generated experiments, which should have an index column
                x_axis = method_to_dsname_to_result_df_list[sample_method][dataset][0]["index"].astype(int)
            ax = plot_metric(method_ds_measure, dataset, x_axis, metric)
            plt.legend()
            outpath = Path(args.output) / (dataset + "-" + metric + ".pdf")
            plt.savefig(str(outpath))


if __name__ == "__main__":
    main()
