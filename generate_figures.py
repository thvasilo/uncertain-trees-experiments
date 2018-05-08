"""
Creates figures for a specific metric from a directory containing subdirs of result csv files.
Will aggregate all the results for every dataset, and plot its mean
value along with its std across experiments.

Will also create a collection of latex tables and corresponding csv output.

Used to plot the output from the skgarden_experiments and moa_experiments scripts.
"""
import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from tabulate import tabulate
from natsort import natsorted

plt.style.use(['seaborn-whitegrid'])

rcParams = matplotlib.rcParams

METHOD_RENAMES = {"OoBConformalApproximate": "CPApproximate", "OoBConformalRegressor": "CPExact",
                  "MondrianForest": "MondrianForest", "PredictiveVarianceRF": "PredictiveVarianceRF",
                  "OnlineQRF": "OnlineQRF"}


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
    parser.add_argument("--dont-create-tables", action="store_true", default=False,
                        help="When given, will not create table files")
    parser.add_argument("--dont-create-figures", action="store_true", default=False,
                        help="When given, will not generate figures.")
    parser.add_argument("--use-tex", action="store_true", default=False,
                        help="When given, will use the Tex engine to create tex for the figures (slower)")
    parser.add_argument("--expected-error", default=0.1,
                        help="The expected error level for the experiment. "
                             "Used to draw the expected error horizontal line and calculate "
                             "the MER deviation.")
    parser.add_argument("--mark-every", default=1, type=int,
                        help="Place a marker on the figures every this many data points.")
    parser.add_argument("--fig-height", type=int, default=6)
    parser.add_argument("--fig-width", type=int, default=6)
    alg_selection = parser.add_mutually_exclusive_group()
    alg_selection.add_argument("--include-only", nargs='+',
                               help="Include only the provided output directories")
    alg_selection.add_argument("--exclude", nargs='+',
                               help="Exclude the provided output directories")

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
        if res_file.match("*.time.csv") or res_file.match("*.pred.csv"):  # Avoid parsing time/pred files
            continue
        base_name = res_file.name[:-6]  # _X.csv is 6 characters
        df = pd.read_csv(res_file)
        res[base_name].append(df)
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


def plot_metric(method_metric_dict, dataset_name, x_axis, metric_name, markevery):
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
    :param markevery: int
        Place a marker every this many points on the lines.
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
        ax.plot(x_axis, mu, label=method, marker=next(marker), markevery=markevery)
        ax.fill_between(x_axis, mu - std_dev, mu + std_dev, alpha=.25, linewidth=0)
    return ax


def sort_nicely(l):
    """ Sort the given list of Path objects in the way that humans expect.
    """
    return sorted(l, key=lambda x: int(x.name) if x.name.isdigit() else x.name)


def create_tables(method_metric_dict, outpath, expected_error):
    """Creates a Latex booktabs table for a specific dataset and metric, for
    each method in the provided dict as well as a csv representation,
    and writes them both to disk. The metrics are aggregated over windows,
    and their mean and stddev are written to disk.
    :param method_metric_dict: {method: {ds_name: measurements_df}}
        Maps each method to dataset names, each one with one aggregate
        measurements df.
        The df has one line per experiment, each column is the measurement
        for one window.
    :param outpath: Path
        A Path object to the output tex file we will create.
    :param expected_error: float
        The expected error rate
    """
    mean_method_to_measurements = OrderedDict()
    std_method_to_measurements = OrderedDict()
    median_method_to_measurements = OrderedDict()
    mean_deviation_method_to_measurements = OrderedDict()

    all_names = []
    for method, ds_metric_dict in method_metric_dict.items():
        ds_names = []
        ds_means = []
        ds_medians = []
        ds_stds = []
        ds_deviation_means = []
        # Get the measurements for the requested data, and calc their stats
        for dataset_name, metric_df in natsorted(ds_metric_dict.items()):
            window_mean = metric_df.mean()
            window_std = metric_df.std()
            mean_window_deviation = abs(window_mean - expected_error).mean()
            # TODO: One DF per dataset, with all methods together, will need to work around iteration order

            ds_outpath = outpath / dataset_name
            ds_outpath.mkdir(exist_ok=True, parents=True)
            window_mean.to_csv(ds_outpath.joinpath("{}_window_means.csv".format(method)))
            window_std.to_csv(ds_outpath.joinpath("{}_window_std.csv".format(method)))

            overall_mean = window_mean.mean()
            overall_median = window_mean.median()
            overall_std = window_std.mean()

            ds_names.append(dataset_name)
            ds_means.append(overall_mean)
            ds_medians.append(overall_median)
            ds_stds.append(overall_std)
            ds_deviation_means.append(mean_window_deviation)

        all_names.append(ds_names)
        mean_method_to_measurements[method] = ds_means
        median_method_to_measurements[method] = ds_medians
        std_method_to_measurements[method] = ds_stds
        mean_deviation_method_to_measurements[method] = ds_deviation_means

    # if outpath.name == "mean_error_rate":
    #     ds_outpath.joinpath("{}_mean_window_deviation.csv".format(method)).write_text(
    #         "method,mean_window_deviation\n{},{}\n".format(method, mean_window_deviation))

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

    mean_aggregate_metric_df = create_df(mean_method_to_measurements)
    std_aggregate_metric_df = create_df(std_method_to_measurements)
    median_aggregate_metric_df = create_df(median_method_to_measurements)

    mean_aggregate_metric_df.to_csv(outpath.with_suffix(".means.csv"))
    std_aggregate_metric_df.to_csv(outpath.with_suffix(".stds.csv"))
    median_aggregate_metric_df.to_csv(outpath.with_suffix(".medians.csv"))

    def create_table_str(df):
        float_format = ".3f" if outpath.name == "mean_error_rate" else ".2f"
        return tabulate(df, headers='keys', tablefmt='latex_booktabs', floatfmt=float_format)

    table_str = create_table_str(mean_aggregate_metric_df)
    std_table_str = create_table_str(std_aggregate_metric_df)
    median_table_str = create_table_str(median_aggregate_metric_df)

    outpath.with_suffix(".means.tex").write_text(table_str)
    outpath.with_suffix(".stds.tex").write_text(std_table_str)
    outpath.with_suffix(".medians.tex").write_text(median_table_str)

    if outpath.name == "mean_error_rate":
        mean_deviation_df = create_df(mean_deviation_method_to_measurements)
        mean_deviation_df.to_csv(outpath.with_suffix(".deviations.csv"))
        mean_deviation_table_str = create_table_str(mean_deviation_df)
        outpath.with_suffix(".deviations.tex").write_text(mean_deviation_table_str)


def main():
    args = parse_args()

    large_text = 28
    small_text = 26
    params = {
        'axes.labelsize': large_text,
        'font.size': small_text,
        'legend.fontsize': large_text,
        'xtick.labelsize': small_text,
        'ytick.labelsize': small_text,
        'text.usetex': args.use_tex,
        'figure.figsize': [args.fig_width, args.fig_height]
    }
    rcParams.update(params)

    input_path = Path(args.input).absolute()
    output_path = Path(args.output).absolute()

    assert output_path.parent != input_path, "Setting output path under input can cause issues, choose another path."
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    # Get all the directories under the input path
    method_dirs = [subpath for subpath in input_path.iterdir() if subpath.is_dir()]
    if args.exclude is not None:
        method_dirs = [subpath for subpath in method_dirs if subpath.name not in args.exclude]
    elif args.include_only is not None:
        method_dirs = [subpath for subpath in method_dirs if subpath.name in args.include_only]

    # Gather the results for each method
    # Format: {method: {ds_name: measurement_df_list}}
    method_to_dsname_to_result_df_list = OrderedDict()
    sorted_dirs = sort_nicely(method_dirs)
    for method_dir in sorted_dirs:
        method_to_dsname_to_result_df_list[method_dir.name] = gather_method_results(method_dir)

    for metric in ["mean error rate", "mean interval size"]:
        # Aggregate the list of result df to a single df per dataset, per method.
        # Format: {method: {ds_name: measurements_df}}
        # Each line in measurements_df is one experiment
        method_ds_metric = OrderedDict()
        for method, ds_to_measurements in method_to_dsname_to_result_df_list.items():
            # TODO: Make it possible to iterate over metrics?
            method_ds_metric[method] = gather_metric(ds_to_measurements, metric)

        # Gather the names of datasets
        ds_names = set()
        for method, ds_name_to_measure in method_ds_metric.items():
            ds_names.update(ds_name_to_measure.keys())

        # All methods should have same x_axis, so just choose one
        sample_method = list(method_to_dsname_to_result_df_list.keys())[0]

        if not args.dont_create_tables:
            table_outpath = Path(args.output) / metric.replace(' ', '_')
            create_tables(method_ds_metric, table_outpath, args.expected_error)

        # Create and save one figure per dataset
        for dataset in ds_names:
            if args.dont_create_figures:
                break
            # From one of the methods, for this dataset, first experiment, get the index, as ints
            try:
                # If this doesn't work, we don't have MOA generated experiments
                x_axis = method_to_dsname_to_result_df_list[sample_method][dataset][0]["learning evaluation instances"].astype(int)
            except KeyError:
                # In which case we should have only Python-generated experiments, which should have an index column
                x_axis = method_to_dsname_to_result_df_list[sample_method][dataset][0]["index"].astype(int)
            ax = plot_metric(method_ds_metric, dataset, x_axis, metric, args.mark_every)
            if metric == "mean error rate":
                ax.axhline(y=args.expected_error, linestyle='dashed', color='grey')
            plt.legend()
            outpath = Path(args.output) / (dataset + "-" + metric.replace(' ', '_') + ".pdf")
            plt.savefig(str(outpath), bbox_inches='tight')


if __name__ == "__main__":
    main()
